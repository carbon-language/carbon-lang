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

import json
import subprocess
import vim

# Change this to the full path if clang-format is not on the path.
binary = 'clang-format'

# Change this to format according to other formatting styles (see 
# clang-format -help)
style = 'LLVM'

# Get the current text.
buf = vim.current.buffer
text = '\n'.join(buf)

# Determine range to format.
cursor = int(vim.eval('line2byte(line("."))+col(".")')) - 2
offset = int(vim.eval('line2byte(' +
                      str(vim.current.range.start + 1) + ')')) - 1
length = int(vim.eval('line2byte(' +
                      str(vim.current.range.end + 2) + ')')) - offset - 2

# Call formatter.
p = subprocess.Popen([binary, '-offset', str(offset), '-length', str(length),
                      '-style', style, '-cursor', str(cursor)],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     stdin=subprocess.PIPE)
stdout, stderr = p.communicate(input=text)

# If successful, replace buffer contents.
if stderr:
  message = stderr.splitlines()[0]
  parts = message.split(' ', 2)
  if len(parts) > 2:
    message = parts[2]
  print 'Formatting failed: %s (total %d warnings, %d errors)' % (
      message, stderr.count('warning:'), stderr.count('error:'))

if not stdout:
  print ('No output from clang-format (crashed?).\n' +
      'Please report to bugs.llvm.org.')
else:
  lines = stdout.split('\n')
  output = json.loads(lines[0])
  lines = lines[1:]
  if '\n'.join(lines) != text:
    for i in range(min(len(buf), len(lines))):
      buf[i] = lines[i]
    for line in lines[len(buf):]:
      buf.append(line)
    del buf[len(lines):]
    vim.command('goto %d' % (output['Cursor'] + 1))
