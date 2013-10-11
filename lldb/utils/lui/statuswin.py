##===-- statuswin.py -----------------------------------------*- Python -*-===##
##
##                     The LLVM Compiler Infrastructure
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===##

import lldb, lldbutil
import cui
import curses

class StatusWin(cui.TextWin):
  def __init__(self, x, y, w, h):
    super(StatusWin, self).__init__(x, y, w)

    self.keys = [#('F1', 'Help', curses.KEY_F1),
                 ('F3', 'Cycle-focus', curses.KEY_F3),
                 ('F10', 'Quit', curses.KEY_F10)]
    text = ''
    for key in self.keys:
      text = text + '{0} {1} '.format(key[0], key[1])
    self.setText(text)

  def handleEvent(self, event):
    if isinstance(event, int):
      pass
    elif isinstance(event, lldb.SBEvent):
      if lldb.SBProcess.EventIsProcessEvent(event):
        state = lldb.SBProcess.GetStateFromEvent(event)
        status = lldbutil.state_type_to_str(state)
        self.win.erase()
        x = self.win.getmaxyx()[1] - len(status) - 1
        self.win.addstr(0, x, status)
    return

