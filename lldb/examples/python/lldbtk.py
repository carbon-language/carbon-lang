#!/usr/bin/python

import lldb
import shlex
import sys
from Tkinter import *
import ttk

def get_item_dictionary_for_sbvalue(v, include_typename):
    '''Given an lldb.SBValue, create an item dictionary for that value and return it
    
    The dictionary must have the following key/value pairs:
        'values'   - must be a list of string values for each column defined in self.get_column_definitions()
        'children' - a boolean value that indicates if an item has children or not
    '''
    name = v.name
    if name is None:
        name = ''
    if include_typename:
        typename = v.type
        if typename is None:
            typename = ''
    value = v.value
    if value is None:
        value = ''
    summary = v.summary
    if summary is None:
        summary = ''
    if include_typename:
        return {   'values' : [name, typename, value, summary], 
                 'children' : v.MightHaveChildren(),
                     'type' : 'SBValue',
                   'object' : v }
    else:
        return {   'values' : [name, value, summary], 
                 'children' : v.MightHaveChildren(),
                     'type' : 'SBValue',
                   'object' : v }
        

def get_item_dictionary_for_process(process):
    id = process.GetProcessID()
    num_threads = process.GetNumThreads()
    value = str(process.GetProcessID())
    summary = process.target.executable.fullpath
    return {   'values' : ['process', value, summary], 
             'children' : num_threads > 0,
                 'type' : 'SBProcess',
               'object' : process }

def get_item_dictionary_for_thread(thread):
    num_frames = thread.GetNumFrames()
    value = '0x%x' % (thread.GetThreadID())
    summary = '%u frames' % (num_frames)
    return {   'values' : ['thread #%u' % (thread.GetIndexID()), value, summary], 
             'children' : num_frames > 0,
                 'type' : 'SBThread',
               'object' : thread }

def get_item_dictionary_for_frame(frame):
    id = frame.GetFrameID()
    value = '0x%16.16x' % (frame.GetPC())
    stream = lldb.SBStream()
    frame.GetDescription(stream)
    summary = stream.GetData().split("`")[1]
    return {   'values' : ['frame #%u' % (id), value, summary], 
             'children' : frame.GetVariables(True, True, True, True).GetSize() > 0,
                 'type' : 'SBFrame',
               'object' : frame }

class ProcessTreeDelegate(object):
    def __init__(self, process):
        self.process = process

    def get_column_definitions(self):
        '''Return an array of column definition dictionaries'''
        return [{ 'id' : '#0'     , 'text' : 'Name'   , 'anchor' : W , 'stretch' : 0 },
                { 'id' : 'value'  , 'text' : 'Value'  , 'anchor' : W , 'stretch' : 0 },
                { 'id' : 'summary', 'text' : 'Summary', 'anchor' : W , 'stretch' : 1 }]

    def get_item_dictionary(self, sbvalue):
        '''Given an lldb.SBValue, create an item dictionary for that value and return it

           The dictionary must have the following key/value pairs:
           'values'   - must be a list of string values for each column defined in self.get_column_definitions()
           'children' - a boolean value that indicates if an item has children or not
        '''

    def get_child_item_dictionaries(self, parent_item_dict):
        '''Given an lldb.SBValue, create an item dictionary for that value and return it'''
        item_dicts = list()
        if parent_item_dict is None:
            # Create root items if parent_item_dict is None
            item_dicts.append(get_item_dictionary_for_process(self.process))
        else:
            # Get children for a specified item given its item dictionary
            item_type = parent_item_dict['type']
            if item_type == 'SBProcess':
                for thread in parent_item_dict['object']:
                    item_dicts.append(get_item_dictionary_for_thread(thread))
            elif item_type == 'SBThread':
                for frame in parent_item_dict['object']:
                    item_dicts.append(get_item_dictionary_for_frame(frame))
            elif item_type == 'SBFrame':
                frame = parent_item_dict['object']
                variables = frame.GetVariables(True, True, True, True)
                n = variables.GetSize()
                for i in range(n):
                    item_dicts.append(get_item_dictionary_for_sbvalue(variables[i], False))
            elif item_type == 'SBValue':
                sbvalue = parent_item_dict['object']
                if sbvalue.IsValid():
                    for i in range(sbvalue.num_children):
                        item_dicts.append(get_item_dictionary_for_sbvalue(sbvalue.GetChildAtIndex(i), False))
        return item_dicts            

class VariableTreeDelegate(object):
    def __init__(self, frame):
        self.frame = frame
    
    def get_column_definitions(self):
        '''Return an array of column definition dictionaries'''
        return [{ 'id' : '#0'     , 'text' : 'Name'   , 'anchor' : W , 'stretch' : 0 },
                { 'id' : 'type'   , 'text' : 'Type'   , 'anchor' : W , 'stretch' : 0 },
                { 'id' : 'value'  , 'text' : 'Value'  , 'anchor' : W , 'stretch' : 0 },
                { 'id' : 'summary', 'text' : 'Summary', 'anchor' : W , 'stretch' : 1 }]
            
    def get_child_item_dictionaries(self, parent_item_dict):
        '''Given an lldb.SBValue, create an item dictionary for that value and return it'''
        item_dicts = list()
        if parent_item_dict is None:
            # Create root items if parent_item_dict is None
            variables = self.frame.GetVariables(True, True, True, True)
            n = variables.GetSize()
            for i in range(n):
                item_dicts.append(get_item_dictionary_for_sbvalue(variables[i], True))
        else:
            # Get children for a specified item given its item dictionary
            sbvalue = parent_item_dict['object']
            if sbvalue.IsValid():
                for i in range(sbvalue.num_children):
                    item_dicts.append(get_item_dictionary_for_sbvalue(sbvalue.GetChildAtIndex(i), True))
        return item_dicts            
            
class DelegateTree(ttk.Frame):
     
    def __init__(self, delegate, title, name):
        ttk.Frame.__init__(self, name=name)
        self.pack(expand=Y, fill=BOTH)
        self.master.title(title)
        self.delegate = delegate
        self.item_id_to_item_dict = dict()
        frame = Frame(self)
        frame.pack(side=TOP, fill=BOTH, expand=Y)
        self._create_treeview(frame)
        self._populate_root()
                     
    def _create_treeview(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(side=TOP, fill=BOTH, expand=Y)
         
        columns_dicts = self.delegate.get_column_definitions()
        column_ids = list()
        for i in range(1,len(columns_dicts)):
            column_ids.append(columns_dicts[i]['id'])
        # create the tree and scrollbars
        self.tree = ttk.Treeview(columns=column_ids)
         
        scroll_bar_v = ttk.Scrollbar(orient=VERTICAL, command= self.tree.yview)
        scroll_bar_h = ttk.Scrollbar(orient=HORIZONTAL, command= self.tree.xview)
        self.tree['yscroll'] = scroll_bar_v.set
        self.tree['xscroll'] = scroll_bar_h.set
         
        # setup column headings and columns properties
        for columns_dict in columns_dicts:
            self.tree.heading(columns_dict['id'], text=columns_dict['text'], anchor=columns_dict['anchor'])
            self.tree.column(columns_dict['id'], stretch=columns_dict['stretch'])
         
        # add tree and scrollbars to frame
        self.tree.grid(in_=frame, row=0, column=0, sticky=NSEW)
        scroll_bar_v.grid(in_=frame, row=0, column=1, sticky=NS)
        scroll_bar_h.grid(in_=frame, row=1, column=0, sticky=EW)
         
        # set frame resizing priorities
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
         
        # action to perform when a node is expanded
        self.tree.bind('<<TreeviewOpen>>', self._update_tree)

    def insert_items(self, parent_id, item_dicts):
        for item_dict in item_dicts:
            values =  item_dict['values']
            item_id = self.tree.insert (parent_id, # root item has an empty name
                                        END, 
                                        text=values[0], 
                                        values=values[1:])
            self.item_id_to_item_dict[item_id] = item_dict
            if item_dict['children']:
                self.tree.insert(item_id, END, text='dummy')
        
    def _populate_root(self):
        # use current directory as root node
        self.insert_items('', self.delegate.get_child_item_dictionaries(None))
    
    def _update_tree(self, event):
        # user expanded a node - build the related directory
        item_id = self.tree.focus()      # the id of the expanded node        
        children = self.tree.get_children (item_id)
        if len(children):
            first_child = children[0]
            # if the node only has a 'dummy' child, remove it and
            # build new directory; skip if the node is already
            # populated
            if self.tree.item(first_child, option='text') == 'dummy':
                self.tree.delete(first_child)
                item_dicts = self.delegate.get_child_item_dictionaries(self.item_id_to_item_dict[item_id])
                self.insert_items(item_id, item_dicts)

@lldb.command("tk-variables")
def tk_variable_display(debugger, command, result, dict):
    sys.argv = ['tk-variables'] # needed for tree creation in TK library as it uses sys.argv...
    target = debugger.GetSelectedTarget()
    if not target:
        print >>result, "invalid target"
        return
    process = target.GetProcess()
    if not process:
        print >>result, "invalid process"
        return
    thread = process.GetSelectedThread()
    if not thread:
        print >>result, "invalid thread"
        return
    frame = thread.GetSelectedFrame()
    if not frame:
        print >>result, "invalid frame"
        return
    # Parse command line args
    command_args = shlex.split(command)
    
    tree = DelegateTree(VariableTreeDelegate(frame), 'Variables', 'lldb-tk-variables')
    tree.mainloop()

@lldb.command("tk-process")
def tk_process_display(debugger, command, result, dict):
    sys.argv = ['tk-process'] # needed for tree creation in TK library as it uses sys.argv...
    target = debugger.GetSelectedTarget()
    if not target:
        print >>result, "invalid target"
        return
    process = target.GetProcess()
    if not process:
        print >>result, "invalid process"
        return
    # Parse command line args
    command_args = shlex.split(command)
    tree = DelegateTree(ProcessTreeDelegate(process), 'Process', 'lldb-tk-process')
    tree.mainloop()

