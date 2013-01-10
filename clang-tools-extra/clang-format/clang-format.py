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

import vim
import subprocess

# Change this to the full path if clang-format is not on the path.
binary = 'clang-format'

# Get the current text.
buf = vim.current.buffer
text = "\n".join(buf)

# Determine range to format.
offset = int(vim.eval('line2byte(' +
                      str(vim.current.range.start + 1) + ')')) - 1
length = int(vim.eval('line2byte(' +
                      str(vim.current.range.end + 2) + ')')) - offset - 2

# Call formatter.
p = subprocess.Popen([binary, '-offset', str(offset), '-length', str(length)],
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
elif stdout != text:
  lines = stdout.split('\n')
  for i in range(min(len(buf), len(lines))):
    buf[i] = lines[i]
  for line in lines[len(buf):]:
    buf.append(line)
  del buf[len(lines):]
