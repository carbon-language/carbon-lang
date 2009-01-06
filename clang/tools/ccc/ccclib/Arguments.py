class Option(object):
    """Root option class."""
    def __init__(self, name):
        self.name = name

    def accept(self, index, arg, it):
        """accept(index, arg, iterator) -> Arg or None
        
        Accept the argument at the given index, returning an Arg, or
        return None if the option does not accept this argument.

        May raise MissingArgumentError.
        """
        abstract

    def __repr__(self):
        return '<%s name=%r>' % (self.__class__.__name__,
                                 self.name)

class FlagOption(Option):
    """An option which takes no arguments."""

    def accept(self, index, arg, it):
        if arg == self.name:
            return Arg(index, self)

class JoinedOption(Option):
    """An option which literally prefixes its argument."""

    def accept(self, index, arg, it):        
        if arg.startswith(self.name):
            return JoinedValueArg(index, self)

class SeparateOption(Option):
    """An option which is followed by its value."""

    def accept(self, index, arg, it):
        if arg == self.name:
            try:
                _,value = it.next()
            except StopIteration:
                raise MissingArgumentError,self
            return SeparateValueArg(index, self)

class MultiArgOption(Option):
    """An option which takes multiple arguments."""

    def __init__(self, name, numArgs):
        assert numArgs > 1
        super(MultiArgOption, self).__init__(name)
        self.numArgs = numArgs

    def accept(self, index, arg, it):
        if arg.startswith(self.name):
            try:
                values = [it.next()[1] for i in range(self.numArgs)]
            except StopIteration:
                raise MissingArgumentError,self
            return MultipleValuesArg(index, self)

class JoinedOrSeparateOption(Option):
    """An option which either literally prefixes its value or is
    followed by an value."""

    def accept(self, index, arg, it):
        if arg.startswith(self.name):
            if len(arg) != len(self.name): # Joined case
                return JoinedValueArg(index, self)
            else:
                try:
                    _,value = it.next()
                except StopIteration:
                    raise MissingArgumentError,self
                return SeparateValueArg(index, self)

class JoinedAndSeparateOption(Option):
    """An option which literally prefixes its value and is followed by
    an value."""

    def accept(self, index, arg, it):
        if arg.startswith(self.name):
            try:
                _,value = it.next()
            except StopIteration:
                raise MissingArgumentError,self
            return JoinedAndSeparateValuesArg(index, self)

###

class Arg(object):
    """Arg - Base class for actual driver arguments."""
    def __init__(self, index, opt):
        self.index = index
        self.opt = opt

    def __repr__(self):
        return '<%s index=%r opt=%r>' % (self.__class__.__name__,
                                         self.index,
                                         self.opt)

    def render(self, args):
        """render(args) -> [str]

        Map the argument into a list of actual program arguments,
        given the source argument array."""
        assert self.opt
        return [self.opt.name]

class ValueArg(Arg):
    """ValueArg - An instance of an option which has an argument."""

    def getValue(self, args):
        abstract

    def setValue(self, args, value):
        abstract

class UnknownArg(ValueArg):
    def __init__(self, index):
        super(UnknownArg, self).__init__(index, None)

    def getValue(self, args):
        return args[self.index]
    
    def setValue(self, args, value):
        args[self.index] = value

    def render(self, args):
        return [args[self.index]]

class JoinedValueArg(ValueArg):
    def getValue(self, args):
        return args[self.index][len(self.opt.name):]

    def setValue(self, args, value):
        assert self.opt.name == args[self.index][:len(self.opt.name)]
        args[self.index] = self.opt.name + value

    def render(self, args):
        return [self.opt.name + self.getValue(args)]

class SeparateValueArg(ValueArg):
    def getValue(self, args):
        return args[self.index+1]

    def setValue(self, args, value):
        args[self.index+1] = value

    def render(self, args):
        return [self.opt.name, self.getValue(args)]

class MultipleValuesArg(Arg):
    def getValues(self, args):
        return args[self.index + 1:self.index + 1 + self.opt.numArgs]

    def setValues(self, args, value):
        assert self.opt.numArgs == len(value)
        args[self.index + 1:self.index + 1 + self.opt.numArgs] = value

    def render(self, args):
        return [self.opt.name] + self.getValues(args)

# FIXME: Man, this is lame. It is only used by -Xarch. Maybe easier to
# just special case?
class JoinedAndSeparateValuesArg(Arg):
    """JoinedAndSeparateValuesArg - An argument with both joined and
    separate values."""

    def getJoinedValue(self, args):
        return args[self.index][len(self.opt.name):]

    def getSeparateValue(self, args):
        return args[self.index+1]

    def setJoinedValue(self, args, value):
        assert self.opt.name == args[self.index][:len(self.opt.name)]
        args[self.index] = self.opt.name + value
        
    def setSeparateValue(self, args, vaue):
        args[self.index+1] = value

    def render(self, args):
        return ([self.opt.name + self.getJoinedValue(args)] + 
                [self.getSeparateValue(args)])

class InputArg(ValueArg):
    """InputArg - An input file (positional) argument."""

    def __init__(self, index):
        super(ValueArg, self).__init__(index, None)

    def getValue(self, args):
        return args[self.index]

    def setValue(self, args, value):
        args[self.index] = value
 
    def render(self, args):
        return [self.getValue(args)]

class DerivedArg(ValueArg):
    """DerivedArg - A synthesized argument which does not correspend
    to the actual input arguments array."""

    def __init__(self, value):
        super(ValueArg, self).__init__(-1, None)
        self.value = value

    def getValue(self, args):
        return self.value

    def setValue(self, args, value):
        raise ValueError,"Cannot call setValue() on a DerivedArg."
    
    def render(self, args):
        return [self.value]

class ArgList:
    """ArgList - Collect an input argv along with a set of parsed Args
    and supporting information."""

    def __init__(self, argv):
        self.argv = list(argv)
        self.args = []

    # Support use as a simple arg list.

    def __iter__(self):
        return iter(self.args)

    def append(self, arg):
        self.args.append(arg)

    # Forwarding methods.

    def getValue(self, arg):
        return arg.getValue(self.argv)

    def getValues(self, arg):
        return arg.getValues(self.argv)

    def getSeparateValue(self, arg):
        return arg.getSeparateValue(self.argv)

    def getJoinedValue(self, arg):
        return arg.getJoinedValue(self.argv)
    
class OptionParser:
    def __init__(self):
        self.options = []
        
    def addOption(self, opt):
        self.options.append(opt)

    def parseArgs(self, argv):
        """
        parseArgs([str]) -> ArgList

        Parse command line into individual option instances.
        """

        iargs = enumerate(argv)
        it = iter(iargs)
        args = ArgList(argv)
        for i,a in it:
            # FIXME: Handle '@'
            if not a: 
                # gcc's handling of empty arguments doesn't make
                # sense, but this is not a common use case. :)
                #
                # We just ignore them here (note that other things may
                # still take them as arguments).
                pass
            elif a[0] == '-' and a != '-':
                args.append(self.lookupOptForArg(i, a, it))
            else:
                args.append(InputArg(i))
        return args
    
    def lookupOptForArg(self, i, arg, it):
        for op in self.options:
            opt = op.accept(i, arg, it)
            if opt is not None:
                return opt
        return UnknownArg(i)

def createOptionParser():
    op = OptionParser()
    
    # Driver driver options
    op.addOption(SeparateOption('-arch'))

    # Misc driver options
    op.addOption(FlagOption('-pass-exit-codes'))
    op.addOption(FlagOption('--help'))
    op.addOption(FlagOption('--target-help'))
    
    op.addOption(FlagOption('-dumpspecs'))
    op.addOption(FlagOption('-dumpversion'))
    op.addOption(FlagOption('-dumpmachine'))
    op.addOption(FlagOption('-print-search-dirs'))
    op.addOption(FlagOption('-print-libgcc-file-name'))
    # FIXME: Hrm, where does this come from? It isn't always true that
    # we take both - and --. For example, gcc --S ... ends up sending
    # -fS to cc1. Investigate.
    op.addOption(FlagOption('--print-libgcc-file-name'))
    op.addOption(JoinedOption('-print-file-name='))
    op.addOption(JoinedOption('-print-prog-name='))
    op.addOption(JoinedOption('--print-prog-name='))
    op.addOption(FlagOption('-print-multi-directory'))
    op.addOption(FlagOption('-print-multi-lib'))
    op.addOption(FlagOption('-print-multi-os-directory'))

    # Hmmm, who really takes this?
    op.addOption(FlagOption('--version'))
    
    # Pipeline control
    op.addOption(FlagOption('-###'))
    op.addOption(FlagOption('-E'))
    op.addOption(FlagOption('-S'))
    op.addOption(FlagOption('-c'))
    op.addOption(FlagOption('-combine'))
    op.addOption(FlagOption('-no-integrated-cpp'))
    op.addOption(FlagOption('-pipe'))
    op.addOption(FlagOption('-save-temps'))
    op.addOption(FlagOption('--save-temps'))
    op.addOption(JoinedOption('-specs='))
    op.addOption(FlagOption('-time'))
    op.addOption(FlagOption('-v'))

    # Input/output stuff
    op.addOption(JoinedOrSeparateOption('-o'))
    op.addOption(JoinedOrSeparateOption('-x'))

    # FIXME: What do these actually do? The documentation is less than
    # clear.
    op.addOption(FlagOption('-ObjC'))
    op.addOption(FlagOption('-ObjC++'))

    # FIXME: Weird, gcc claims this here in help but I'm not sure why;
    # perhaps interaction with preprocessor? Investigate.
    op.addOption(JoinedOption('-std='))
    op.addOption(JoinedOrSeparateOption('--sysroot'))

    # Version control
    op.addOption(JoinedOrSeparateOption('-B'))
    op.addOption(JoinedOrSeparateOption('-V'))
    op.addOption(JoinedOrSeparateOption('-b'))

    # Blanket pass-through options.
    
    op.addOption(JoinedOption('-Wa,'))
    op.addOption(SeparateOption('-Xassembler'))

    op.addOption(JoinedOption('-Wp,'))
    op.addOption(SeparateOption('-Xpreprocessor'))

    op.addOption(JoinedOption('-Wl,'))
    op.addOption(SeparateOption('-Xlinker'))

    ####
    # Bring on the random garbage.

    op.addOption(FlagOption('-MD'))
    op.addOption(FlagOption('-MP'))
    op.addOption(FlagOption('-MM'))
    op.addOption(JoinedOrSeparateOption('-MF'))
    op.addOption(JoinedOrSeparateOption('-MT'))
    op.addOption(FlagOption('-undef'))

    op.addOption(FlagOption('-w'))
    op.addOption(JoinedOrSeparateOption('-allowable_client'))
    op.addOption(JoinedOrSeparateOption('-client_name'))
    op.addOption(JoinedOrSeparateOption('-compatibility_version'))
    op.addOption(JoinedOrSeparateOption('-current_version'))
    op.addOption(JoinedOrSeparateOption('-exported_symbols_list'))
    op.addOption(JoinedOrSeparateOption('-idirafter'))
    op.addOption(JoinedOrSeparateOption('-iquote'))
    op.addOption(JoinedOrSeparateOption('-isysroot'))
    op.addOption(JoinedOrSeparateOption('-keep_private_externs'))
    op.addOption(JoinedOrSeparateOption('-seg1addr'))
    op.addOption(JoinedOrSeparateOption('-segprot'))
    op.addOption(JoinedOrSeparateOption('-sub_library'))
    op.addOption(JoinedOrSeparateOption('-sub_umbrella'))
    op.addOption(JoinedOrSeparateOption('-umbrella'))
    op.addOption(JoinedOrSeparateOption('-undefined'))
    op.addOption(JoinedOrSeparateOption('-unexported_symbols_list'))
    op.addOption(JoinedOrSeparateOption('-weak_framework'))
    op.addOption(JoinedOption('-headerpad_max_install_names'))
    op.addOption(FlagOption('-twolevel_namespace'))
    op.addOption(FlagOption('-prebind'))
    op.addOption(FlagOption('-prebind_all_twolevel_modules'))
    op.addOption(FlagOption('-single_module'))
    op.addOption(FlagOption('-nomultidefs'))
    op.addOption(FlagOption('-nostdlib'))
    op.addOption(FlagOption('-nostdinc'))
    op.addOption(FlagOption('-static'))
    op.addOption(FlagOption('-shared'))
    op.addOption(FlagOption('-C'))
    op.addOption(FlagOption('-CC'))
    op.addOption(FlagOption('-R'))
    op.addOption(FlagOption('-P'))
    op.addOption(FlagOption('-all_load'))
    op.addOption(FlagOption('--constant-cfstrings'))
    op.addOption(FlagOption('-traditional'))
    op.addOption(FlagOption('--traditional'))
    op.addOption(FlagOption('-no_dead_strip_inits_and_terms'))
    op.addOption(MultiArgOption('-sectalign', numArgs=3))
    op.addOption(MultiArgOption('-sectcreate', numArgs=3))
    op.addOption(MultiArgOption('-sectorder', numArgs=3))

    # I dunno why these don't end up working when joined. Maybe
    # because of translation?
    op.addOption(SeparateOption('-filelist'))
    op.addOption(SeparateOption('-framework'))
    op.addOption(SeparateOption('-install_name'))
    op.addOption(SeparateOption('-seg_addr_table'))
    op.addOption(SeparateOption('-seg_addr_table_filename'))

    # Where are these coming from? I can't find them...
    op.addOption(JoinedOrSeparateOption('-e')) # Gets forwarded to linker
    op.addOption(JoinedOrSeparateOption('-r'))

    # Is this actually declared anywhere? I can only find it in a
    # spec. :(
    op.addOption(FlagOption('-pg'))

    doNotReallySupport = 1
    if doNotReallySupport:
        # Archaic gcc option.
        op.addOption(FlagOption('-cpp-precomp'))
        op.addOption(FlagOption('-no-cpp-precomp'))

    # C options for testing

    op.addOption(JoinedOrSeparateOption('-include'))
    op.addOption(JoinedOrSeparateOption('-A'))
    op.addOption(JoinedOrSeparateOption('-D'))
    op.addOption(JoinedOrSeparateOption('-F'))
    op.addOption(JoinedOrSeparateOption('-I'))
    op.addOption(JoinedOrSeparateOption('-L'))
    op.addOption(JoinedOrSeparateOption('-U'))
    op.addOption(JoinedOrSeparateOption('-l'))
    op.addOption(JoinedOrSeparateOption('-u'))

    # FIXME: What is going on here? '-X' goes to linker, and -X ... goes nowhere?
    op.addOption(FlagOption('-X'))
    # Not exactly sure how to decompose this. I split out -Xarch_
    # because we need to recognize that in the driver driver part.
    # FIXME: Man, this is lame it needs its own option.
    op.addOption(JoinedAndSeparateOption('-Xarch_'))
    op.addOption(JoinedOption('-X'))

    # The driver needs to know about this flag.
    op.addOption(FlagOption('-fsyntax-only'))

    # FIXME: Wrong?
    # FIXME: What to do about the ambiguity of options like
    # -dumpspecs? How is this handled in gcc?
    op.addOption(JoinedOption('-d'))
    op.addOption(JoinedOption('-g'))
    op.addOption(JoinedOption('-f'))
    op.addOption(JoinedOption('-m'))
    op.addOption(JoinedOption('-i'))
    op.addOption(JoinedOption('-O'))
    op.addOption(JoinedOption('-W'))
    # FIXME: Weird. This option isn't really separate, --param=a=b
    # works. An alias somewhere?
    op.addOption(SeparateOption('--param'))

    # FIXME: What is this? Seems to do something on Linux. I think
    # only one is valid, but have a log that uses both.
    op.addOption(FlagOption('-pthread'))
    op.addOption(FlagOption('-pthreads'))

    return op
