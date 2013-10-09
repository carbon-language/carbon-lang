import cui
import curses
import lldb, lldbutil

class BreakWin(cui.TitledWin):
  def __init__(self, driver, x, y, w, h):
    super(BreakWin, self).__init__(x, y, w, h, 'Breakpoints')
    self.win.scrollok(1)
    super(BreakWin, self).draw()
    self.driver = driver

  def handleEvent(self, event):
    if isinstance(event, lldb.SBEvent):
      if lldb.SBBreakpoint.EventIsBreakpointEvent(event):
        self.update()

  def update(self):
    target = self.driver.getTarget()
    if not target.IsValid():
      return
    self.win.addstr(0, 0, '')
    for i in range(0, target.GetNumBreakpoints()):
      bp = target.GetBreakpointAtIndex(i)
      if bp.IsInternal():
        continue
      text = lldbutil.get_description(bp)
      self.win.addstr(text)
      self.win.addstr('\n')
