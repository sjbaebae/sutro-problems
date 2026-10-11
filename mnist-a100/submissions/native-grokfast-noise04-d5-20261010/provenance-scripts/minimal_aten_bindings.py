"""Bounded source-to-source rewrite for archived Torch C++ bindings.

This does not compile, load, or mutate any native source. The caller must qualify
its rewritten extension for ABI, numerical, graph/reset and cold-build behavior.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

HEADERS = (
    '#include <ATen/core/Tensor.h>',
    '#include <ATen/ops/empty.h>',
    '#include <ATen/ops/empty_like.h>',
    '#include <torch/csrc/utils/pybind.h>',
)
ALIASES = {
    'torch::Tensor': 'at::Tensor',
    'torch::empty_like': 'at::empty_like',
    'torch::empty': 'at::empty',
    'torch::kFloat32': 'at::kFloat',
    'torch::kInt64': 'at::kLong',
    'torch::kInt32': 'at::kInt',
    'torch::kBFloat16': 'at::kBFloat16',
}
# Protect comments, normal string/character literals, and C++ raw literals.
_PROTECTED = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|R"(?P<delimiter>[^\s()\\]{0,16})\([\s\S]*?\)(?P=delimiter)"'
    r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
)
_HEAVY = re.compile(
    r'(?m)^[ \t]*#[ \t]*include[ \t]*[<"]torch/(?:types|extension)\.h[>"][ \t]*(?://[^\n]*)?(?:\n|$)'
)
_NAMESPACE = re.compile(r'\btorch\s*::\s*[A-Za-z_]\w*')
_ALIAS = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in sorted(ALIASES, key=len, reverse=True)) + r')\b')


def _code_only(source: str) -> str:
    """Keep byte positions/newlines, hide comments and literals for token checks."""
    return _PROTECTED.sub(lambda m: ''.join('\n' if c == '\n' else ' ' for c in m.group()), source)


def rewrite_bindings_source(source: str) -> str:
    """Return minimal-header ATen bindings; reject unsupported torch:: tokens.

    Only complete heavy include lines and the listed namespace aliases change.
    CUDA headers, validation, graph orchestration, pybind module declarations and
    every other character are preserved. Already rewritten text is idempotent.
    """
    if not isinstance(source, str):
        raise TypeError('source must be str')
    code = _code_only(source)
    namespace_tokens = [m.group() for m in _NAMESPACE.finditer(code)]
    unsupported = sorted(set(namespace_tokens) - set(ALIASES))
    if unsupported:
        raise ValueError('unsupported torch:: usages: ' + ', '.join(unsupported))
    includes = [m for m in _HEAVY.finditer(source)
                if code[m.start():m.end()].lstrip().startswith('#')]
    if not includes:
        if namespace_tokens or not all(h in source for h in HEADERS):
            raise ValueError('expected torch/types.h or torch/extension.h include')
        return source
    edits: list[tuple[int, int, str]] = []
    header_block = '\n'.join(h for h in HEADERS if h not in source) + '\n'
    for i, match in enumerate(includes):
        edits.append((match.start(), match.end(), header_block if i == 0 else ''))
    for match in _ALIAS.finditer(code):
        edits.append((match.start(), match.end(), ALIASES[match.group()]))
    edits.sort()
    cursor = 0
    pieces = []
    for start, end, replacement in edits:
        if start < cursor:
            raise AssertionError('overlapping rewrite edits')
        pieces.append(source[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(source[cursor:])
    result = ''.join(pieces)
    result_code = _code_only(result)
    remaining_heavy = [m for m in _HEAVY.finditer(result)
                       if result_code[m.start():m.end()].lstrip().startswith('#')]
    if remaining_heavy or _NAMESPACE.search(result_code):
        raise AssertionError('heavy headers or unhandled torch:: code remain')
    if not all(h in result for h in HEADERS):
        raise AssertionError('minimal headers incomplete')
    # Reconstruction checks every untouched interval, including CUDA code.
    output_cursor = 0
    input_cursor = 0
    for start, end, replacement in edits:
        unchanged = source[input_cursor:start]
        assert result[output_cursor:output_cursor + len(unchanged)] == unchanged
        output_cursor += len(unchanged)
        assert result[output_cursor:output_cursor + len(replacement)] == replacement
        output_cursor += len(replacement)
        input_cursor = end
    assert result[output_cursor:] == source[input_cursor:]
    assert source.count('PYBIND11_MODULE') == result.count('PYBIND11_MODULE')
    return result


minimal_aten_bindings = rewrite_bindings_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('use a separate output file; this helper never edits original bindings')
    with args.input.open(newline='') as handle:
        source = handle.read()
    result = rewrite_bindings_source(source)
    with args.output.open('w', newline='') as handle:
        handle.write(result)


if __name__ == '__main__':
    main()
