import cui
import curses
import lldb

class History(object):
  def __init__(self):
    self.data = {}
    self.pos = 0
    self.tempEntry = ''

  def previous(self, curr):
    if self.pos == len(self.data):
      self.tempEntry = curr

    if self.pos < 0:
      return ''
    if self.pos == 0:
      self.pos -= 1
      return ''
    if self.pos > 0:
      self.pos -= 1
      return self.data[self.pos]

  def next(self):
    if self.pos < len(self.data):
      self.pos += 1

    if self.pos < len(self.data):
      return self.data[self.pos]
    elif self.tempEntry != '':
      return self.tempEntry
    else:
      return ''

  def add(self, c):
    self.tempEntry = ''
    self.pos = len(self.data)
    if self.pos == 0 or self.data[self.pos-1] != c:
      self.data[self.pos] = c
      self.pos += 1

class CommandWin(cui.TitledWin):
  def __init__(self, driver, x, y, w, h):
    super(CommandWin, self).__init__(x, y, w, h, "Commands")
    self.command = ""

    driver.setSize(w, h)

    self.win.scrollok(1)

    self.driver = driver
    self.history = History()

    def enterCallback(content):
      self.handleCommand(content)
    def tabCompleteCallback(content):
      pass # TODO: implement

    self.startline = self.win.getmaxyx()[0]-2

    self.el = cui.CursesEditLine(self.win, self.history, enterCallback, tabCompleteCallback)
    self.el.prompt = self.driver.getPrompt()
    self.el.showPrompt(self.startline, 0)

  def handleCommand(self, cmd):
     # enter!
    self.win.scroll(1) # TODO: scroll more for longer commands
    if cmd == '':
      cmd = self.history.previous('')
    elif cmd in ('q', 'quit'):
      self.driver.terminate()
      return

    self.history.add(cmd)
    ret = self.driver.handleCommand(cmd)
    if ret.Succeeded():
      out = ret.GetOutput()
      attr = curses.A_NORMAL
    else:
      out = ret.GetError()
      attr = curses.color_pair(3) # red on black
    self.win.addstr(self.startline, 0, out + '\n', attr)
    self.win.scroll(1)
    self.el.showPrompt(self.startline, 0)

  def handleEvent(self, event):
    if isinstance(event, int):
      if event == curses.ascii.EOT and self.el.content == '':
        # When the command is empty, treat CTRL-D as EOF.
        self.driver.terminate()
        return
      self.el.handleEvent(event)

