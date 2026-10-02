"""Shared command-line and environment helpers for the three examples."""
from contextlib import contextmanager
from pathlib import Path
import os
from urllib.parse import urlsplit


def positive_int(value):
    from argparse import ArgumentTypeError
    try:
        number = int(value)
    except ValueError as exc:
        raise ArgumentTypeError('expected a positive integer') from exc
    if number <= 0:
        raise ArgumentTypeError('expected a positive integer')
    return number


def add_api_arguments(parser):
    parser.add_argument('--model', default=os.getenv('EVODIVERSE_MODEL', 'DeepSeek-V3.2'))
    parser.add_argument('--check', action='store_true',
                        help='Check dependencies and inputs without calling a model API.')


def configure_api(parser, model):
    key = os.getenv('EVODIVERSE_API_KEY')
    base_url = os.getenv('EVODIVERSE_BASE_URL')
    if not key or not base_url:
        parser.error('Set EVODIVERSE_API_KEY and EVODIVERSE_BASE_URL in your environment.')
    parsed = urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        parser.error('EVODIVERSE_BASE_URL must be an HTTP(S) API base URL.')
    os.environ['EVODIVERSE_MODEL'] = model
    return key, base_url


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
