#!/usr/bin/env python

"""
                     The LLVM Compiler Infrastructure

 This file is distributed under the University of Illinois Open Source
 License. See LICENSE.TXT for details.

Configuration options for lldbtest.py set by dotest.py during initialization
"""

import curses
import lldbcurses
import sys
import test_results

class Curses(test_results.ResultsFormatter):
    """Receives live results from tests that are running and reports them to the terminal in a curses GUI"""

    def __init__(self, out_file, options):
        # Initialize the parent
        super(Curses, self).__init__(out_file, options)
        self.using_terminal = True
        self.have_curses = True
        self.initialize_event = None
        self.jobs = [None] * 64
        self.job_tests = [None] * 64
        self.results = list()
        self.saved_first_responder = None
        try:
            self.main_window = lldbcurses.intialize_curses()
            self.main_window.refresh()
            self.job_panel = None
            self.results_panel = None
            self.status_panel = None
            self.info_panel = None
        except:
            self.have_curses = False
            lldbcurses.terminate_curses()
            self.using_terminal = False
            print "Unexpected error:", sys.exc_info()[0]
            raise
            
        
        self.line_dict = dict()
        #self.events_file = open("/tmp/events.txt", "w")
        # self.formatters = list()
        # if tee_results_formatter:
        #     self.formatters.append(tee_results_formatter)

    def status_to_short_str(self, status):
        if status == 'success':
            return '.'
        elif status == 'failure':
            return 'F'
        elif status == 'unexpected_success':
            return '?'
        elif status == 'expected_failure':
            return 'X'
        elif status == 'skip':
            return 'S'
        elif status == 'error':
            return 'E'
        else:
            return status

    def handle_info_panel_key(self, window, key):
        window.resign_first_responder(remove_from_parent=True, new_first_responder=self.saved_first_responder)
        window.hide()        
        self.saved_first_responder = None
        self.main_window.refresh()
        return True

    def handle_job_panel_key(self, window, key):
        return False

    def handle_result_panel_key(self, window, key):
        if key == ord('\r') or key == ord('\n') or key == curses.KEY_ENTER:
            selected_idx = self.results_panel.get_selected_idx()
            if selected_idx >= 0 and selected_idx < len(self.results):
                if self.info_panel is None:
                    info_frame = self.results_panel.get_contained_rect(top_inset=10, left_inset=10, right_inset=10, height=30)
                    self.info_panel = lldbcurses.BoxedPanel(info_frame, "Result Details", delegate=self.handle_info_panel_key)
                    self.info_panel.top()
                else:
                    self.info_panel.show()
                
                self.saved_first_responder = self.main_window.first_responder
                self.main_window.set_first_responder(self.info_panel)
                test_start = self.results[selected_idx][0]
                test_result = self.results[selected_idx][1]
                self.info_panel.set_line(0, "File: %s" % (test_start['test_filename']))
                self.info_panel.set_line(1, "Test: %s.%s" % (test_start['test_class'], test_start['test_name']))
                self.info_panel.set_line(2, "Time: %s" % (test_result['elapsed_time']))
                self.info_panel.set_line(3, "Status: %s" % (test_result['status']))
        elif key == curses.KEY_HOME:
            self.results_panel.scroll_begin()
        elif key == curses.KEY_END:
            self.results_panel.scroll_end()
        elif key == curses.KEY_UP:
            self.results_panel.select_prev()
        elif key == curses.KEY_DOWN:
            self.results_panel.select_next()
        else:
            return False
        self.main_window.refresh()
        
    def handle_event(self, test_event):
        with self.lock:
            super(Curses, self).handle_event(test_event)
            # for formatter in self.formatters:
            #     formatter.process_event(test_event)
            if self.have_curses:
                worker_index = -1
                if 'worker_index' in test_event:
                    worker_index = test_event['worker_index']
                if 'event' in test_event:
                    check_for_one_key = True
                    #print >>self.events_file, str(test_event)
                    event = test_event['event']
                    if event == 'test_start':
                        name = test_event['test_class'] + '.' + test_event['test_name']
                        self.job_tests[worker_index] = test_event
                        if 'pid' in test_event:
                            line = 'pid: %5d ' % (test_event['pid']) + name
                        else:
                            line = name
                        self.job_panel.set_line(worker_index, line)
                        self.main_window.refresh()
                    elif event == 'test_result':
                        status = test_event['status']
                        self.status_panel.increment_status(status)
                        if 'pid' in test_event:
                            line = 'pid: %5d ' % (test_event['pid'])
                        else:
                            line = ''
                        self.job_panel.set_line(worker_index, line)
                        # if status != 'success' and status != 'skip' and status != 'expect_failure':
                        name = test_event['test_class'] + '.' + test_event['test_name']
                        elapsed_time = test_event['event_time'] - self.job_tests[worker_index]['event_time']
                        self.results_panel.append_line('%s (%6.2f sec) %s' % (self.status_to_short_str(status), elapsed_time, name))
                        self.main_window.refresh()
                        # Append the result pairs
                        test_event['elapsed_time'] = elapsed_time
                        self.results.append([self.job_tests[worker_index], test_event])
                        self.job_tests[worker_index] = ''
                    elif event == 'job_begin':
                        self.jobs[worker_index] = test_event
                        if 'pid' in test_event:
                            line = 'pid: %5d ' % (test_event['pid'])
                        else:
                            line = ''
                        self.job_panel.set_line(worker_index, line)
                    elif event == 'job_end':
                        self.jobs[worker_index] = ''
                        self.job_panel.set_line(worker_index, '')
                    elif event == 'initialize':
                        self.initialize_event = test_event
                        num_jobs = test_event['worker_count']
                        job_frame = self.main_window.get_contained_rect(height=num_jobs+2)
                        results_frame = self.main_window.get_contained_rect(top_inset=num_jobs+2, bottom_inset=1)
                        status_frame = self.main_window.get_contained_rect(height=1, top_inset=self.main_window.get_size().h-1)
                        self.job_panel = lldbcurses.BoxedPanel(frame=job_frame, title="Jobs", delegate=self.handle_job_panel_key)
                        self.results_panel = lldbcurses.BoxedPanel(frame=results_frame, title="Results", delegate=self.handle_result_panel_key)
                        self.status_panel = lldbcurses.StatusPanel(frame=status_frame)
                        
                        self.main_window.add_child(self.job_panel)
                        self.main_window.add_child(self.results_panel)
                        self.main_window.add_child(self.status_panel)
                        self.main_window.set_first_responder(self.results_panel)
                        
                        self.status_panel.add_status_item(name="success", title="Success", format="%u", width=20, value=0, update=False)
                        self.status_panel.add_status_item(name="failure", title="Failure", format="%u", width=20, value=0, update=False)
                        self.status_panel.add_status_item(name="error", title="Error", format="%u", width=20, value=0, update=False)
                        self.status_panel.add_status_item(name="skip", title="Skipped", format="%u", width=20, value=0, update=True)
                        self.status_panel.add_status_item(name="expected_failure", title="Expected Failure", format="%u", width=30, value=0, update=False)
                        self.status_panel.add_status_item(name="unexpected_success", title="Unexpected Success", format="%u", width=30, value=0, update=False)
                        self.main_window.refresh()
                    elif event == 'terminate':
                        self.main_window.key_event_loop()
                        lldbcurses.terminate_curses()
                        check_for_one_key = False
                        self.using_terminal = False
                        # Check for 1 keypress with no delay 
                    
                    # Check for 1 keypress with no delay
                    if check_for_one_key:
                        self.main_window.key_event_loop(0, 1) 
                        
