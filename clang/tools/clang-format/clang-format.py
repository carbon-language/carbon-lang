# This file is a minimal clang-format vim-integration. To install:
# - Change 'binary' if clang-format is not on the path (see below).
# - Add to your .vimrc:
#
#   map <C-I> :pyf <path-to-this-file>/clang-format.py<CR>
#   imap <C-I> <ESC>:pyf <path-to-this-file>/clang-format.py<CR>i
#
# The first line enables clang-format for NORMAL and VISUAL mode, the second
# line adds support for INSERT mode. Change "C-I" to another binding if you
# need clang-format on a different key (C-I stands for Ctrl+i).
#
# With this integration you can press the bound key and clang-format will
# format the current line in NORMAL and INSERT mode or the selected region in
# VISUAL mode. The line or region is extended to the next bigger syntactic
# entity.
#
# It operates on the current, potentially unsaved buffer and does not create
# or save any files. To revert a formatting, just undo.

import difflib
import json
import subprocess
import sys
import vim

# Change this to the full path if clang-format is not on the path.
binary = 'clang-format'

# Change this to format according to other formatting styles. See the output of
# 'clang-format --help' for a list of supported styles. The default looks for
# a '.clang-format' or '_clang-format' file to indicate the style that should be
# used.
style = 'file'

def main():
  # Get the current text.
  buf = vim.current.buffer
  text = '\n'.join(buf)

  # Determine range to format.
  lines = '%s:%s' % (vim.current.range.start + 1, vim.current.range.end + 1)

  # Determine the cursor position.
  cursor = int(vim.eval('line2byte(line("."))+col(".")')) - 2
  if cursor < 0:
    print 'Couldn\'t determine cursor position. Is your file empty?'
    return

  # Avoid flashing an ugly, ugly cmd prompt on Windows when invoking clang-format.
  startupinfo = None
  if sys.platform.startswith('win32'):
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE

  # Call formatter.
  command = [binary, '-lines', lines, '-style', style, '-cursor', str(cursor)]
  if vim.current.buffer.name:
    command.extend(['-assume-filename', vim.current.buffer.name])
  p = subprocess.Popen(command,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.PIPE, startupinfo=startupinfo)
  stdout, stderr = p.communicate(input=text)

  # If successful, replace buffer contents.
  if stderr:
    print stderr

  if not stdout:
    print ('No output from clang-format (crashed?).\n' +
        'Please report to bugs.llvm.org.')
  else:
    lines = stdout.split('\n')
    output = json.loads(lines[0])
    lines = lines[1:]
    sequence = difflib.SequenceMatcher(None, vim.current.buffer, lines)
    for op in reversed(sequence.get_opcodes()):
      if op[0] is not 'equal':
        vim.current.buffer[op[1]:op[2]] = lines[op[3]:op[4]]
    vim.command('goto %d' % (output['Cursor'] + 1))

main()
