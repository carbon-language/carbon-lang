##===-- breakwin.py ------------------------------------------*- Python -*-===##
##
##                     The LLVM Compiler Infrastructure
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===##

import cui
import curses
import lldb, lldbutil

class BreakWin(cui.ListWin):
  def __init__(self, driver, x, y, w, h):
    super(BreakWin, self).__init__(x, y, w, h)
    self.driver = driver
    self.update()

  def handleEvent(self, event):
    if isinstance(event, lldb.SBEvent):
      if lldb.SBBreakpoint.EventIsBreakpointEvent(event):
        self.update()
    if isinstance(event, int):
      if event == ord('d'):
        self.deleteSelected()
    super(BreakWin, self).handleEvent(event)

  def deleteSelected(self):
    if self.getSelected() == -1:
      return
    target = self.driver.getTarget()
    if not target.IsValid():
      return
    bp = target.GetBreakpointAtIndex(self.getSelected())
    target.BreakpointDelete(bp.id)

  def update(self):
    target = self.driver.getTarget()
    if not target.IsValid():
      return
    selected = self.getSelected()
    self.clearItems()
    for i in range(0, target.GetNumBreakpoints()):
      bp = target.GetBreakpointAtIndex(i)
      if bp.IsInternal():
        continue
      text = lldbutil.get_description(bp)
      self.addItem(text)
    self.setSelected(selected)
