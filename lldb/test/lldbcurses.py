import time 
import curses, curses.panel

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "(x=%u, y=%u)" % (self.x, self.y)
    
    def is_valid_coordinate(self):
        return self.x >= 0 and self.y >= 0
        
class Size(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "(w=%u, h=%u)" % (self.w, self.h)

class Rect(object):
    def __init__(self, x=0, y=0, w=0, h=0):
        self.origin = Point(x, y)
        self.size = Size(w, h)
        
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "{ %s, %s }" % (str(self.origin), str(self.size))

    def get_min_x(self):
        return self.origin.x

    def get_max_x(self):
        return self.origin.x + self.size.w

    def get_min_y(self):
        return self.origin.y

    def get_max_y(self):
        return self.origin.y + self.size.h
    
    def contains_point(self, pt):
        if pt.x < self.get_max_x():
            if pt.y < self.get_max_y():
                if pt.x >= self.get_min_y():
                    return pt.y >= self.get_min_y()
        return False

class Window(object):
    def __init__(self, window):
        self.window = window

    def point_in_window(self, pt):
        size = self.get_size()
        return pt.x >= 0 and pt.x < size.w and pt.y >= 0 and pt.y < size.h

    def addstr(self, pt, str):
        try:
            self.window.addstr(pt.y, pt.x, str)
        except:
            pass

    def addnstr(self, pt, str, n):
        try:
            self.window.addnstr(pt.y, pt.x, str, n)
        except:
            pass

    def box(self):
        self.window.box()

    def get_contained_rect(self, top_inset=0, bottom_inset=0, left_inset=0, right_inset=0, height=-1, width=-1):
        '''Get a rectangle based on the top "height" lines of this window'''
        rect = self.get_frame()
        x = rect.origin.x + left_inset
        y = rect.origin.y + top_inset
        if height == -1:
            h = rect.size.h - (top_inset + bottom_inset)
        else:
            h = height
        if width == -1:
            w = rect.size.w - (left_inset + right_inset)
        else:
            w = width
        return Rect (x = x, y = y, w = w, h = h)

    def erase(self):
        self.window.erase()

    def get_frame(self):
        position = self.get_position()
        size = self.get_size()
        return Rect(x=position.x, y=position.y, w=size.w, h=size.h)

    def get_position(self):
        (y, x) = self.window.getbegyx()
        return Point(x, y)

    def get_size(self):
        (y, x) = self.window.getmaxyx()
        return Size(w=x, h=y)
    
    def refresh(self):
        curses.panel.update_panels()
        return self.window.refresh()
        
    def resize(self, size):
        return window.resize(size.h, size.w)
    
class Panel(Window):
    def __init__(self, frame):
        window = curses.newwin(frame.size.h,frame.size.w, frame.origin.y, frame.origin.x)
        super(Panel, self).__init__(window)
        self.panel = curses.panel.new_panel(window)

    def top(self):
        self.panel.top()
    
    def set_position(self, pt):
        self.panel.move(pt.y, pt.x)
    
    def slide_position(self, pt):
        new_position = self.get_position()
        new_position.x = new_position.x + pt.x
        new_position.y = new_position.y + pt.y
        self.set_position(new_position)

class BoxedPanel(Panel):
    def __init__(self, frame, title):
        super(BoxedPanel, self).__init__(frame)
        self.title = title
        self.lines = list()
        self.first_visible_idx = 0
        self.update()

    def get_usable_width(self):
        '''Valid usable width is 0 to (width - 3) since the left and right lines display the box around 
           this frame and we skip a leading space'''
        w = self.get_size().w
        if w > 3:
            return w-3
        else:
            return 0
    
    def get_usable_height(self):
        '''Valid line indexes are 0 to (height - 2) since the top and bottom lines display the box around this frame.'''
        h = self.get_size().h
        if h > 2:
            return h-2
        else:
            return 0

    def get_point_for_line(self, global_line_idx):
        '''Returns the point to use when displaying a line whose index is "line_idx"'''
        line_idx = global_line_idx - self.first_visible_idx
        num_lines = self.get_usable_height()
        if line_idx < num_lines:
            return Point(x=2, y=1+line_idx)
        else:
            return Point(x=-1, y=-1) # return an invalid coordinate if the line index isn't valid
        
    def set_title (self, title, update=True):
        self.title = title
        if update:
            self.update()

    def _adjust_first_visible_line(self):
        num_lines = len(self.lines)
        max_visible_lines = self.get_usable_height()
        if (num_lines - self.first_visible_idx) > max_visible_lines:
            self.first_visible_idx = num_lines - max_visible_lines
        
    def append_line(self, s, update=True):
        self.lines.append(s)
        self._adjust_first_visible_line()
        if update:
            self.update()

    def set_line(self, line_idx, s, update=True):
        '''Sets a line "line_idx" within the boxed panel to be "s"'''
        if line_idx < 0:
            return
        while line_idx >= len(self.lines):
            self.lines.append('')
        self.lines[line_idx] = s
        self._adjust_first_visible_line()
        if update:
            self.update()
    
    def update(self):
        self.erase()
        self.box()
        if self.title:
            self.addstr(Point(x=2, y=0), ' ' + self.title + ' ')
        max_width = self.get_usable_width()
        for line_idx in range(self.first_visible_idx, len(self.lines)):
            pt = self.get_point_for_line(line_idx)
            if pt.is_valid_coordinate():
                self.addnstr(pt, self.lines[line_idx], max_width)
            else:
                return

class StatusPanel(Panel):
    def __init__(self, frame):
        super(StatusPanel, self).__init__(frame)
        self.status_items = list()
        self.status_dicts = dict()
        self.next_status_x = 1
    
    def add_status_item(self, name, title, format, width, value, update=True):
        status_item_dict = { 'name': name,
                             'title' : title,
                             'width' : width,
                             'format' : format,
                             'value' : value,
                             'x' : self.next_status_x }
        index = len(self.status_items)
        self.status_items.append(status_item_dict)
        self.status_dicts[name] = index
        self.next_status_x += width + 2;
        if update:
            self.update()

    def increment_status(self, name, update=True):
        if name in self.status_dicts:
            status_item_idx = self.status_dicts[name]
            status_item_dict = self.status_items[status_item_idx]
            status_item_dict['value'] = status_item_dict['value'] + 1
        if update:
            self.update()
    
    def update_status(self, name, value, update=True):
        if name in self.status_dicts:
            status_item_idx = self.status_dicts[name]
            status_item_dict = self.status_items[status_item_idx]
            status_item_dict['value'] = status_item_dict['format'] % (value)
        if update:
            self.update()
    def update(self):
        self.erase();
        for status_item_dict in self.status_items:
            self.addnstr(Point(x=status_item_dict['x'], y=0), '%s: %s' % (status_item_dict['title'], status_item_dict['value']), status_item_dict['width'])

stdscr = None

def intialize_curses():
    global stdscr
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(1)
    try:
        curses.start_color()
    except:
        pass
    return Window(stdscr)

def terminate_curses():
    global stdscr
    if stdscr:
        stdscr.keypad(0)
    curses.echo()
    curses.nocbreak()
    curses.endwin()

