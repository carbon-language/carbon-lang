#
# modify-lldb-python.py
#
# This script modifies the lldb module (which was automatically generated via
# running swig) to support iteration and/or equality operations for certain lldb
# objects, implements truth value testing for certain lldb objects, and adds a
# global variable 'debugger_unique_id' which is initialized to 0.
#
# It also calls SBDebugger.Initialize() to initialize the lldb debugger
# subsystem.
#

import sys, re, StringIO

if len (sys.argv) != 2:
    output_name = "./lldb.py"
else:
    output_name = sys.argv[1] + "/lldb.py"

# print "output_name is '" + output_name + "'"

#
# lldb_iter() should appear before our first SB* class definition.
#
lldb_iter_def = '''
# ===================================
# Iterator for lldb container objects
# ===================================
def lldb_iter(obj, getsize, getelem):
    """A generator adaptor to support iteration for lldb container objects."""
    size = getattr(obj, getsize)
    elem = getattr(obj, getelem)
    for i in range(size()):
        yield elem(i)

# ==============================================================================
# The modify-python-lldb.py script is responsible for post-processing this SWIG-
# generated lldb.py module.  It is responsible for adding the above lldb_iter()
# function definition as well as the supports, in the following, for iteration
# protocol: __iter__, rich comparison methods: __eq__ and __ne__, truth value
# testing (and built-in operation bool()): __nonzero__, and built-in function
# len(): __len__.
# ==============================================================================
'''

# This supports the iteration protocol.
iter_def = "    def __iter__(self): return lldb_iter(self, '%s', '%s')"
module_iter = "    def module_iter(self): return lldb_iter(self, '%s', '%s')"
breakpoint_iter = "    def breakpoint_iter(self): return lldb_iter(self, '%s', '%s')"

# Called to implement the built-in function len().
# Eligible objects are those containers with unambiguous iteration support.
len_def = "    def __len__(self): return self.%s()"

# This supports the rich comparison methods of __eq__ and __ne__.
eq_def = "    def __eq__(self, other): return isinstance(other, %s) and %s"
ne_def = "    def __ne__(self, other): return not self.__eq__(other)"

# Called to implement truth value testing and the built-in operation bool();
# should return False or True, or their integer equivalents 0 or 1.
# Delegate to self.IsValid() if it is defined for the current lldb object.
nonzero_def = "    def __nonzero__(self): return self.IsValid()"

#
# This dictionary defines a mapping from classname to (getsize, getelem) tuple.
#
d = { 'SBBreakpoint':  ('GetNumLocations',   'GetLocationAtIndex'),
      'SBCompileUnit': ('GetNumLineEntries', 'GetLineEntryAtIndex'),
      'SBDebugger':    ('GetNumTargets',     'GetTargetAtIndex'),
      'SBModule':      ('GetNumSymbols',     'GetSymbolAtIndex'),
      'SBProcess':     ('GetNumThreads',     'GetThreadAtIndex'),
      'SBThread':      ('GetNumFrames',      'GetFrameAtIndex'),

      'SBInstructionList':   ('GetSize', 'GetInstructionAtIndex'),
      'SBStringList':        ('GetSize', 'GetStringAtIndex',),
      'SBSymbolContextList': ('GetSize', 'GetContextAtIndex'),
      'SBValueList':         ('GetSize',  'GetValueAtIndex'),

      'SBType':  ('GetNumberChildren', 'GetChildAtIndex'),
      'SBValue': ('GetNumChildren',    'GetChildAtIndex'),

      # SBTarget needs special processing, see below.
      'SBTarget': {'module':     ('GetNumModules', 'GetModuleAtIndex'),
                   'breakpoint': ('GetNumBreakpoints', 'GetBreakpointAtIndex')
                   }
      }

#
# This dictionary defines a mapping from classname to equality method name(s).
#
e = { 'SBBreakpoint': ['GetID'],
      'SBFileSpec':   ['GetFilename', 'GetDirectory'],
      'SBModule':     ['GetFileSpec', 'GetUUIDString']
      }

def list_to_frag(list):
    """Transform a list to equality program fragment.

    For example, ['GetID'] is transformed to 'self.GetID() == other.GetID()',
    and ['GetFilename', 'GetDirectory'] to 'self.GetFilename() == other.GetFilename()
    and self.GetDirectory() == other.GetDirectory()'.
    """
    if not list:
        raise Exception("list should be non-empty")
    frag = StringIO.StringIO()
    for i in range(len(list)):
        if i > 0:
            frag.write(" and ")
        frag.write("self.{0}() == other.{0}()".format(list[i]))
    return frag.getvalue()

# The new content will have the iteration protocol defined for our lldb objects.
new_content = StringIO.StringIO()

with open(output_name, 'r') as f_in:
    content = f_in.read()

# The pattern for recognizing the beginning of an SB class definition.
class_pattern = re.compile("^class (SB.*)\(_object\):$")

# The pattern for recognizing the beginning of the __init__ method definition.
init_pattern = re.compile("^    def __init__\(self, \*args\):")

# The pattern for recognizing the beginning of the IsValid method definition.
isvalid_pattern = re.compile("^    def IsValid\(\*args\):")

# These define the states of our finite state machine.
NORMAL = 0
DEFINING_ITERATOR = 1
DEFINING_EQUALITY = 2

# The lldb_iter_def only needs to be inserted once.
lldb_iter_defined = False;

# Our FSM begins its life in the NORMAL state, and transitions to the
# DEFINING_ITERATOR and/or DEFINING_EQUALITY state whenever it encounters the
# beginning of certain class definitions, see dictionaries 'd' and 'e' above.
#
# Note that the two states DEFINING_ITERATOR and DEFINING_EQUALITY are
# orthogonal in that our FSM can be in one, the other, or both states at the
# same time.  During such time, the FSM is eagerly searching for the __init__
# method definition in order to insert the appropriate method(s) into the lldb
# module.
#
# The FSM, in all possible states, also checks the current input for IsValid()
# definition, and inserts a __nonzero__() method definition to implement truth
# value testing and the built-in operation bool().
state = NORMAL
for line in content.splitlines():
    if state == NORMAL:
        match = class_pattern.search(line)
        # Inserts the lldb_iter() definition before the first class definition.
        if not lldb_iter_defined and match:
            print >> new_content, lldb_iter_def
            lldb_iter_defined = True

        # If we are at the beginning of the class definitions, prepare to
        # transition to the DEFINING_ITERATOR/DEFINING_EQUALITY state for the
        # right class names.
        if match:
            cls = match.group(1)
            if cls in d:
                # Adding support for iteration for the matched SB class.
                state = (state | DEFINING_ITERATOR)
            if cls in e:
                # Adding support for eq and ne for the matched SB class.
                state = (state | DEFINING_EQUALITY)
    elif state > NORMAL:
        match = init_pattern.search(line)
        if match:
            # We found the beginning of the __init__ method definition.
            # This is a good spot to insert the iter and/or eq-ne support.
            #
            # But note that SBTarget has two types of iterations.
            if cls == "SBTarget":
                print >> new_content, module_iter % (d[cls]['module'])
                print >> new_content, breakpoint_iter % (d[cls]['breakpoint'])
            else:
                if (state & DEFINING_ITERATOR):
                    print >> new_content, iter_def % d[cls]
                    print >> new_content, len_def % d[cls][0]
                if (state & DEFINING_EQUALITY):
                    print >> new_content, eq_def % (cls, list_to_frag(e[cls]))
                    print >> new_content, ne_def

            # Next state will be NORMAL.
            state = NORMAL

    # Look for 'def IsValid(*args):', and once located, add implementation
    # of truth value testing for this object by delegation.
    if isvalid_pattern.search(line):
        print >> new_content, nonzero_def

    # Pass the original line of content to new_content.
    print >> new_content, line
    
with open(output_name, 'w') as f_out:
    f_out.write(new_content.getvalue())
    f_out.write("debugger_unique_id = 0\n")
    f_out.write("SBDebugger.Initialize()\n")
