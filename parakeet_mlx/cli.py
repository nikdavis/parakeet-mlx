import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import typer
from mlx.core import bfloat16, float32
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from typing_extensions import Annotated

from parakeet_mlx import AlignedResult, AlignedSentence, AlignedToken, from_pretrained

app = typer.Typer(no_args_is_help=True)


# helpers
def format_timestamp(
    seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    assert seconds >= 0
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def to_txt(result: AlignedResult) -> str:
    """Format transcription result as plain text."""
    return result.text.strip()


def to_srt(result: AlignedResult, highlight_words: bool = False) -> str:
    """
    Format transcription result as an SRT file.
    """
    srt_content = []
    entry_index = 1
    if highlight_words:
        for sentence in result.sentences:
            for i, token in enumerate(sentence.tokens):
                start_time = format_timestamp(token.start, decimal_marker=",")
                end_time = format_timestamp(
                    token.end
                    if token == sentence.tokens[-1]
                    else sentence.tokens[i + 1].start,
                    decimal_marker=",",
                )

                text = ""
                for j, inner_token in enumerate(sentence.tokens):
                    if i == j:
                        text += inner_token.text.replace(
                            inner_token.text.strip(),
                            f"<u>{inner_token.text.strip()}</u>",
                        )
                    else:
                        text += inner_token.text
                text.strip()

                srt_content.append(f"{entry_index}")
                srt_content.append(f"{start_time} --> {end_time}")
                srt_content.append(text)
                srt_content.append("")
                entry_index += 1
    else:
        for sentence in result.sentences:
            start_time = format_timestamp(sentence.start, decimal_marker=",")
            end_time = format_timestamp(sentence.end, decimal_marker=",")
            text = sentence.text.strip()

            srt_content.append(f"{entry_index}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
            entry_index += 1

    return "\n".join(srt_content)


def to_vtt(result: AlignedResult, highlight_words: bool = False) -> str:
    """
    Format transcription result as a VTT file.
    """
    vtt_content = ["WEBVTT", ""]
    if highlight_words:
        for sentence in result.sentences:
            for i, token in enumerate(sentence.tokens):
                start_time = format_timestamp(token.start, decimal_marker=".")
                end_time = format_timestamp(
                    token.end
                    if token == sentence.tokens[-1]
                    else sentence.tokens[i + 1].start,
                    decimal_marker=".",
                )

                text_line = ""
                for j, inner_token in enumerate(sentence.tokens):
                    if i == j:
                        text_line += inner_token.text.replace(
                            inner_token.text.strip(),
                            f"<b>{inner_token.text.strip()}</b>",
                        )
                    else:
                        text_line += inner_token.text
                text_line = text_line.strip()

                vtt_content.append(f"{start_time} --> {end_time}")
                vtt_content.append(text_line)
                vtt_content.append("")
    else:
        for sentence in result.sentences:
            start_time = format_timestamp(sentence.start, decimal_marker=".")
            end_time = format_timestamp(sentence.end, decimal_marker=".")
            text_line = sentence.text.strip()

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text_line)
            vtt_content.append("")

    return "\n".join(vtt_content)


def _aligned_token_to_dict(token: AlignedToken) -> Dict[str, Any]:
    return {
        "text": token.text,
        "start": round(token.start, 3),
        "end": round(token.end, 3),
        "duration": round(token.duration, 3),
    }


def _aligned_sentence_to_dict(sentence: AlignedSentence) -> Dict[str, Any]:
    return {
        "text": sentence.text,
        "start": round(sentence.start, 3),
        "end": round(sentence.end, 3),
        "duration": round(sentence.duration, 3),
        "tokens": [_aligned_token_to_dict(token) for token in sentence.tokens],
    }


def to_json(result: AlignedResult) -> str:
    output_dict = {
        "text": result.text,
        "sentences": [
            _aligned_sentence_to_dict(sentence) for sentence in result.sentences
        ],
    }
    return json.dumps(output_dict, indent=2, ensure_ascii=False)


@app.command("transcribe")
def transcribe(
    audios: Annotated[
        List[Path],
        typer.Argument(
            help="Files to transcribe",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model: Annotated[
        str, typer.Option(help="Hugging Face repository of model to use")
    ] = "mlx-community/parakeet-tdt-0.6b-v2",
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save transcriptions")
    ] = Path("."),
    output_format: Annotated[
        str, typer.Option(help="Format for output files (txt, srt, vtt, json, all)")
    ] = "srt",
    output_template: Annotated[
        str,
        typer.Option(
            help="Template for output filenames, e.g. '{filename}_{date}_{index}'"
        ),
    ] = "{filename}",
    highlight_words: Annotated[
        bool,
        typer.Option(help="Underline/timestamp each word as it is spoken in srt/vtt"),
    ] = False,
    chunk_duration: Annotated[
        float,
        typer.Option(
            help="Chunking duration in seconds for long audio, 0 to disable chunking."
        ),
    ] = 60 * 2,
    overlap_duration: Annotated[
        float, typer.Option(help="Overlap duration in seconds if using chunking")
    ] = 15,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print out process and debug messages"),
    ] = False,
    fp32: Annotated[
        bool, typer.Option("--fp32/--bf16", help="Use FP32 precision")
    ] = False,
    to_stdout: Annotated[
        bool,
        typer.Option("--to-stdout", help="Print transcription to stdout (best for 'txt' format) instead of writing a file."),
    ] = False,
):
    """
    Transcribe audio files using Parakeet MLX models.
    """
    if verbose:
        print(f"Loading model: [bold cyan]{model}[/bold cyan]...")

    try:
        loaded_model = from_pretrained(model, dtype=bfloat16 if not fp32 else float32)
        if verbose:
            print("[green]Model loaded successfully.[/green]")
    except Exception as e:
        print(f"[bold red]Error loading model {model}:[/bold red] {e}")
        raise typer.Exit(code=1)

    if not (to_stdout and output_format.lower() == "txt"): # Only skip dir creation if ONLY txt output is going to stdout
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[bold red]Error creating output directory {output_dir}:[/bold red] {e}")
            raise typer.Exit(code=1)

    if verbose:
        print(f"Output directory: [bold cyan]{output_dir.resolve()}[/bold cyan]")
        print(f"Output format(s): [bold cyan]{output_format}[/bold cyan]")
        if output_format in ["srt", "vtt", "all"] and highlight_words:
            print("Highlight words: [bold cyan]Enabled[/bold cyan]")
        if to_stdout:
            print("Output to stdout: [bold cyan]Enabled[/bold cyan] (primarily for 'txt' format)")


    formatters = {
        "txt": to_txt,
        "srt": lambda r: to_srt(r, highlight_words=highlight_words),
        "vtt": lambda r: to_vtt(r, highlight_words=highlight_words),
        "json": to_json,
    }

    formats_to_generate = []
    if output_format == "all":
        formats_to_generate = list(formatters.keys())
    elif output_format in formatters:
        formats_to_generate = [output_format]
    else:
        print(
            f"[bold red]Error: Invalid output format '{output_format}'. Choose from {list(formatters.keys()) + ['all']}.[/bold red]"
        )
        raise typer.Exit(code=1)

    total_files = len(audios)
    if verbose:
        print(f"Transcribing {total_files} file(s)...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True if not verbose else False, # Keep progress if verbose for easier log reading
    ) as progress:
        main_task_id = progress.add_task("Overall Progress...", total=total_files)

        for i, audio_path in enumerate(audios):
            progress.update(main_task_id, description=f"Processing [cyan]{audio_path.name}[/cyan] ({i+1}/{total_files})")

            chunk_processing_task_id = None # For nested progress bar for chunks

            def chunk_callback_for_progress(current_chunk_pos, total_chunk_audio_len):
                nonlocal chunk_processing_task_id
                if chunk_processing_task_id is None:
                     # Task description includes audio file name for clarity if multiple files run in verbose
                    task_description = f"Chunking [magenta]{audio_path.name}[/magenta]"
                    chunk_processing_task_id = progress.add_task(task_description, total=total_chunk_audio_len)
                progress.update(chunk_processing_task_id, completed=current_chunk_pos)


            try:
                result: AlignedResult = loaded_model.transcribe(
                    str(audio_path), # Ensure path is string for older audiofile versions
                    dtype=bfloat16 if not fp32 else float32,
                    chunk_duration=chunk_duration if chunk_duration != 0 else None,
                    overlap_duration=overlap_duration,
                    chunk_callback=chunk_callback_for_progress if not verbose else None, # Use rich progress for chunking if not verbose
                )

                if chunk_processing_task_id is not None: # Remove chunk task after it's done
                    progress.remove_task(chunk_processing_task_id)


                # Suppress per-sentence verbose output if --to-stdout is used for 'txt'
                # to avoid mixing with the final clean stdout output.
                if verbose and not (to_stdout and "txt" in formats_to_generate):
                    print(f"\n[bold]Transcription for {audio_path.name}:[/bold]")
                    for sentence in result.sentences:
                        start, end, text = sentence.start, sentence.end, sentence.text
                        line = f"[blue][{format_timestamp(start)} --> {format_timestamp(end)}][/blue] {text.strip()}"
                        print(line)
                    print("") # Add a newline after per-sentence output

                base_filename = audio_path.stem
                template_vars = {
                    "filename": base_filename,
                    "date": datetime.datetime.now().strftime("%Y%m%d"),
                    "index": str(i + 1),
                }
                output_basename = output_template.format(**template_vars)

                for fmt in formats_to_generate:
                    formatter = formatters[fmt]
                    output_content = formatter(result)

                    if to_stdout and fmt == "txt":
                        # Use regular print for stdout to avoid rich console markup issues if piped
                        import sys
                        sys.stdout.write(output_content + "\n")
                        sys.stdout.flush()
                        if verbose:
                            # Use rich.print for console messages
                            print(f"[green]Printed '{fmt.upper()}' output to stdout for {audio_path.name}.[/green]")
                        continue # Skip file writing for 'txt' format if it went to stdout

                    output_filename = f"{output_basename}.{fmt}"
                    output_filepath = output_dir / output_filename

                    try:
                        with open(output_filepath, "w", encoding="utf-8") as f:
                            f.write(output_content)
                        if verbose:
                            print(
                                f"[green]Saved {fmt.upper()}:[/green] {output_filepath.resolve()}"
                            )
                    except Exception as e:
                        print(
                            f"[bold red]Error writing output file {output_filepath}:[/bold red] {e}"
                        )
            except Exception as e:
                if chunk_processing_task_id is not None:
                    progress.remove_task(chunk_processing_task_id)
                print(f"[bold red]Error transcribing file {audio_path}:[/bold red] {e}")
                # Optionally: raise e # if you want to halt on first error

            progress.update(main_task_id, advance=1)


    # Final message
    if to_stdout and "txt" in formats_to_generate and len(formats_to_generate) == 1 and total_files == 1:
        # If only one txt file was processed to stdout, the "Transcription complete" message might be redundant
        # or could be tailored. For now, we'll keep it.
        pass

    if not (to_stdout and output_format.lower() == "txt" and len(formats_to_generate) ==1 ):
         print(
            f"\n[bold green]Transcription complete.[/bold green] Outputs saved in '{output_dir.resolve()}' (unless printed to stdout)."
        )
    elif verbose : # if it was to_stdout and txt only
        print(f"\n[bold green]Transcription to stdout complete.[/bold green]")


if __name__ == "__main__":
    app()
