#!/usr/bin/python

import sys
import time

class ProgressBar(object):
    """ProgressBar class holds the options of the progress bar.
    The options are:
        start   State from which start the progress. For example, if start is 
                5 and the end is 10, the progress of this state is 50%
        end     State in which the progress has terminated.
        width   --
        fill    String to use for "filled" used to represent the progress
        blank   String to use for "filled" used to represent remaining space.
        format  Format
        incremental
    """
    light_block = unichr(0x2591).encode("utf-8")
    solid_block = unichr(0x2588).encode("utf-8")
    solid_right_arrow = unichr(0x25BA).encode("utf-8")
    
    def __init__(self, 
                 start=0, 
                 end=10, 
                 width=12, 
                 fill=unichr(0x25C9).encode("utf-8"), 
                 blank=unichr(0x25CC).encode("utf-8"), 
                 marker=unichr(0x25CE).encode("utf-8"), 
                 format='[%(fill)s%(marker)s%(blank)s] %(progress)s%%', 
                 incremental=True):
        super(ProgressBar, self).__init__()

        self.start = start
        self.end = end
        self.width = width
        self.fill = fill
        self.blank = blank
        self.marker = marker
        self.format = format
        self.incremental = incremental
        self.step = 100 / float(width) #fix
        self.reset()

    def __add__(self, increment):
        increment = self._get_progress(increment)
        if 100 > self.progress + increment:
            self.progress += increment
        else:
            self.progress = 100
        return self

    def complete(self):
        self.progress = 100
        return self

    def __str__(self):
        progressed = int(self.progress / self.step) #fix
        fill = progressed * self.fill
        blank = (self.width - progressed) * self.blank
        return self.format % {'fill': fill, 'blank': blank, 'marker': self.marker, 'progress': int(self.progress)}

    __repr__ = __str__

    def _get_progress(self, increment):
        return float(increment * 100) / self.end

    def reset(self):
        """Resets the current progress to the start point"""
        self.progress = self._get_progress(self.start)
        return self


class AnimatedProgressBar(ProgressBar):
    """Extends ProgressBar to allow you to use it straighforward on a script.
    Accepts an extra keyword argument named `stdout` (by default use sys.stdout)
    and may be any file-object to which send the progress status.
    """
    def __init__(self, 
                 start=0, 
                 end=10, 
                 width=12, 
                 fill=unichr(0x25C9).encode("utf-8"), 
                 blank=unichr(0x25CC).encode("utf-8"), 
                 marker=unichr(0x25CE).encode("utf-8"), 
                 format='[%(fill)s%(marker)s%(blank)s] %(progress)s%%', 
                 incremental=True,
                 stdout=sys.stdout):
        super(AnimatedProgressBar, self).__init__(start,end,width,fill,blank,marker,format,incremental)
        self.stdout = stdout

    def show_progress(self):
        if hasattr(self.stdout, 'isatty') and self.stdout.isatty():
            self.stdout.write('\r')
        else:
            self.stdout.write('\n')
        self.stdout.write(str(self))
        self.stdout.flush()


if __name__ == '__main__':
    p = AnimatedProgressBar(end=200, width=200)

    while True:
        p + 5
        p.show_progress()
        time.sleep(0.3)
        if p.progress == 100:
            break
    print #new line