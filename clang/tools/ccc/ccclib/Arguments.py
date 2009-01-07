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

# Dummy options

class InputOption(Option):
    def __init__(self):
        super(InputOption, self).__init__('<input>')

    def accept(self):
        raise RuntimeError,"accept() should never be used on InputOption instance."

class UnknownOption(Option):
    def __init__(self):
        super(UnknownOption, self).__init__('<unknown>')

    def accept(self):
        raise RuntimeError,"accept() should never be used on UnknownOption instance."        

# Normal options

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
        assert opt is not None
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

class PositionalArg(ValueArg):
    """PositionalArg - A simple positional argument."""

    def getValue(self, args):
        return args.getInputString(self.index)

    def render(self, args):
        return [args.getInputString(self.index)]

class JoinedValueArg(ValueArg):
    """JoinedValueArg - A single value argument where the value is
    joined (suffixed) to the option."""

    def getValue(self, args):
        return args.getInputString(self.index)[len(self.opt.name):]

    def render(self, args):
        return [self.opt.name + self.getValue(args)]

class SeparateValueArg(ValueArg):
    """SeparateValueArg - A single value argument where the value
    follows the option in the argument vector."""

    def getValue(self, args):
        return args.getInputString(self.index, offset=1)

    def render(self, args):
        return [self.opt.name, self.getValue(args)]

class MultipleValuesArg(Arg):
    """MultipleValuesArg - An argument with multiple values which
    follow the option in the argument vector."""

    # FIXME: Should we unify this with SeparateValueArg?

    def getValues(self, args):
        return [args.getInputString(self.index, offset=1+i)
                for i in range(self.opt.numArgs)]

    def render(self, args):
        return [self.opt.name] + self.getValues(args)

# FIXME: Man, this is lame. It is only used by -Xarch. Maybe easier to
# just special case?
class JoinedAndSeparateValuesArg(Arg):
    """JoinedAndSeparateValuesArg - An argument with both joined and
    separate values."""

    def getJoinedValue(self, args):
        return args.getInputString(self.index)[len(self.opt.name):]

    def getSeparateValue(self, args):
        return args.getInputString(self.index, offset=1)

    def render(self, args):
        return ([self.opt.name + self.getJoinedValue(args)] + 
                [self.getSeparateValue(args)])

class DerivedArg(ValueArg):
    """DerivedArg - A synthesized argument which does not correspend
    to an item in the argument vector."""

    def __init__(self, value):
        # FIXME: The UnknownOption() here is a total hack so we can
        # rely on arg.opt not being nil. Ok for now since DerivedArg
        # is dying.
        super(DerivedArg, self).__init__(-1, UnknownOption())
        self.value = value

    def getValue(self, args):
        return self.value

    def render(self, args):
        return [self.value]

class ArgList:
    """ArgList - Collect an input argument vector along with a set of parsed Args
    and supporting information."""

    def __init__(self, argv):
        self.argv = list(argv)
        self.args = []
        self.lastArgs = {}

    def getLastArg(self, option):
        return self.lastArgs.get(option)

    def getInputString(self, index, offset=0):
        return self.argv[index + offset]

    # Support use as a simple arg list.

    def __iter__(self):
        return iter(self.args)

    def append(self, arg):
        self.args.append(arg)
        self.lastArgs[arg.opt] = arg

    # Forwarding methods.
    #
    # FIXME: Clean this up once restructuring is done.

    def render(self, arg):
        return arg.render(self)

    def getValue(self, arg):
        return arg.getValue(self)

    def getValues(self, arg):
        return arg.getValues(self)

    def getSeparateValue(self, arg):
        return arg.getSeparateValue(self)

    def getJoinedValue(self, arg):
        return arg.getJoinedValue(self)

###
    
class OptionParser:
    def __init__(self):
        self.options = []
        self.inputOption = InputOption()
        self.unknownOption = UnknownOption()

        # Driver driver options
        self.archOption = self.addOption(SeparateOption('-arch'))

        # Misc driver options
        self.addOption(FlagOption('-pass-exit-codes'))
        self.addOption(FlagOption('--help'))
        self.addOption(FlagOption('--target-help'))

        self.dumpspecsOption = self.addOption(FlagOption('-dumpspecs'))
        self.dumpversionOption = self.addOption(FlagOption('-dumpversion'))
        self.dumpmachineOption = self.addOption(FlagOption('-dumpmachine'))
        self.printSearchDirsOption = self.addOption(FlagOption('-print-search-dirs'))
        self.printLibgccFilenameOption = self.addOption(FlagOption('-print-libgcc-file-name'))
        # FIXME: Hrm, where does this come from? It isn't always true that
        # we take both - and --. For example, gcc --S ... ends up sending
        # -fS to cc1. Investigate.
        #
        # FIXME: Need to implement some form of alias support inside
        # getLastOption to handle this.
        self.printLibgccFileNameOption2 = self.addOption(FlagOption('--print-libgcc-file-name'))
        self.printFileNameOption = self.addOption(JoinedOption('-print-file-name='))
        self.printProgNameOption = self.addOption(JoinedOption('-print-prog-name='))
        self.printProgNameOption2 = self.addOption(JoinedOption('--print-prog-name='))
        self.printMultiDirectoryOption = self.addOption(FlagOption('-print-multi-directory'))
        self.printMultiLibOption = self.addOption(FlagOption('-print-multi-lib'))
        self.addOption(FlagOption('-print-multi-os-directory'))

        # Hmmm, who really takes this?
        self.addOption(FlagOption('--version'))

        # Pipeline control
        self.hashHashHashOption = self.addOption(FlagOption('-###'))
        self.EOption = self.addOption(FlagOption('-E'))
        self.SOption = self.addOption(FlagOption('-S'))
        self.cOption = self.addOption(FlagOption('-c'))
        self.combineOption = self.addOption(FlagOption('-combine'))
        self.noIntegratedCPPOption = self.addOption(FlagOption('-no-integrated-cpp'))
        self.pipeOption = self.addOption(FlagOption('-pipe'))
        self.saveTempsOption = self.addOption(FlagOption('-save-temps'))
        self.saveTempsOption2 = self.addOption(FlagOption('--save-temps'))
        self.addOption(JoinedOption('-specs='))
        self.addOption(FlagOption('-time'))
        self.addOption(FlagOption('-v'))

        # Input/output stuff
        self.oOption = self.addOption(JoinedOrSeparateOption('-o'))
        self.xOption = self.addOption(JoinedOrSeparateOption('-x'))

        # FIXME: What do these actually do? The documentation is less than
        # clear.
        self.addOption(FlagOption('-ObjC'))
        self.addOption(FlagOption('-ObjC++'))

        # FIXME: Weird, gcc claims this here in help but I'm not sure why;
        # perhaps interaction with preprocessor? Investigate.
        self.addOption(JoinedOption('-std='))
        self.addOption(JoinedOrSeparateOption('--sysroot'))

        # Version control
        self.addOption(JoinedOrSeparateOption('-B'))
        self.addOption(JoinedOrSeparateOption('-V'))
        self.addOption(JoinedOrSeparateOption('-b'))

        # Blanket pass-through options.

        self.addOption(JoinedOption('-Wa,'))
        self.addOption(SeparateOption('-Xassembler'))

        self.addOption(JoinedOption('-Wp,'))
        self.addOption(SeparateOption('-Xpreprocessor'))

        self.addOption(JoinedOption('-Wl,'))
        self.addOption(SeparateOption('-Xlinker'))

        ####
        # Bring on the random garbage.

        self.addOption(FlagOption('-MD'))
        self.addOption(FlagOption('-MP'))
        self.addOption(FlagOption('-MM'))
        self.addOption(JoinedOrSeparateOption('-MF'))
        self.addOption(JoinedOrSeparateOption('-MT'))
        self.addOption(FlagOption('-undef'))

        self.addOption(FlagOption('-w'))
        self.addOption(JoinedOrSeparateOption('-allowable_client'))
        self.addOption(JoinedOrSeparateOption('-client_name'))
        self.addOption(JoinedOrSeparateOption('-compatibility_version'))
        self.addOption(JoinedOrSeparateOption('-current_version'))
        self.addOption(JoinedOrSeparateOption('-exported_symbols_list'))
        self.addOption(JoinedOrSeparateOption('-idirafter'))
        self.addOption(JoinedOrSeparateOption('-iquote'))
        self.addOption(JoinedOrSeparateOption('-isysroot'))
        self.addOption(JoinedOrSeparateOption('-keep_private_externs'))
        self.addOption(JoinedOrSeparateOption('-seg1addr'))
        self.addOption(JoinedOrSeparateOption('-segprot'))
        self.addOption(JoinedOrSeparateOption('-sub_library'))
        self.addOption(JoinedOrSeparateOption('-sub_umbrella'))
        self.addOption(JoinedOrSeparateOption('-umbrella'))
        self.addOption(JoinedOrSeparateOption('-undefined'))
        self.addOption(JoinedOrSeparateOption('-unexported_symbols_list'))
        self.addOption(JoinedOrSeparateOption('-weak_framework'))
        self.addOption(JoinedOption('-headerpad_max_install_names'))
        self.addOption(FlagOption('-twolevel_namespace'))
        self.addOption(FlagOption('-prebind'))
        self.addOption(FlagOption('-prebind_all_twolevel_modules'))
        self.addOption(FlagOption('-single_module'))
        self.addOption(FlagOption('-nomultidefs'))
        self.addOption(FlagOption('-nostdlib'))
        self.addOption(FlagOption('-nostdinc'))
        self.addOption(FlagOption('-static'))
        self.addOption(FlagOption('-shared'))
        self.addOption(FlagOption('-C'))
        self.addOption(FlagOption('-CC'))
        self.addOption(FlagOption('-R'))
        self.addOption(FlagOption('-P'))
        self.addOption(FlagOption('-all_load'))
        self.addOption(FlagOption('--constant-cfstrings'))
        self.addOption(FlagOption('-traditional'))
        self.addOption(FlagOption('--traditional'))
        self.addOption(FlagOption('-no_dead_strip_inits_and_terms'))
        self.addOption(MultiArgOption('-sectalign', numArgs=3))
        self.addOption(MultiArgOption('-sectcreate', numArgs=3))
        self.addOption(MultiArgOption('-sectorder', numArgs=3))

        # I dunno why these don't end up working when joined. Maybe
        # because of translation?
        self.filelistOption = self.addOption(SeparateOption('-filelist'))
        self.addOption(SeparateOption('-framework'))
        self.addOption(SeparateOption('-install_name'))
        self.addOption(SeparateOption('-seg_addr_table'))
        self.addOption(SeparateOption('-seg_addr_table_filename'))

        # Where are these coming from? I can't find them...
        self.addOption(JoinedOrSeparateOption('-e')) # Gets forwarded to linker
        self.addOption(JoinedOrSeparateOption('-r'))

        # Is this actually declared anywhere? I can only find it in a
        # spec. :(
        self.addOption(FlagOption('-pg'))

        doNotReallySupport = 1
        if doNotReallySupport:
            # Archaic gcc option.
            self.addOption(FlagOption('-cpp-precomp'))
            self.addOption(FlagOption('-no-cpp-precomp'))

        # C options for testing

        self.addOption(JoinedOrSeparateOption('-include'))
        self.addOption(JoinedOrSeparateOption('-A'))
        self.addOption(JoinedOrSeparateOption('-D'))
        self.addOption(JoinedOrSeparateOption('-F'))
        self.addOption(JoinedOrSeparateOption('-I'))
        self.addOption(JoinedOrSeparateOption('-L'))
        self.addOption(JoinedOrSeparateOption('-U'))
        self.addOption(JoinedOrSeparateOption('-l'))
        self.addOption(JoinedOrSeparateOption('-u'))

        # FIXME: What is going on here? '-X' goes to linker, and -X ... goes nowhere?
        self.addOption(FlagOption('-X'))
        # Not exactly sure how to decompose this. I split out -Xarch_
        # because we need to recognize that in the driver driver part.
        # FIXME: Man, this is lame it needs its own option.
        self.XarchOption = self.addOption(JoinedAndSeparateOption('-Xarch_'))
        self.addOption(JoinedOption('-X'))

        # The driver needs to know about this flag.
        self.syntaxOnlyOption = self.addOption(FlagOption('-fsyntax-only'))

        # FIXME: Wrong?
        # FIXME: What to do about the ambiguity of options like
        # -dumpspecs? How is this handled in gcc?
        self.addOption(JoinedOption('-d'))
        self.addOption(JoinedOption('-g'))
        self.addOption(JoinedOption('-f'))
        self.addOption(JoinedOption('-m'))
        self.addOption(JoinedOption('-i'))
        self.addOption(JoinedOption('-O'))
        self.addOption(JoinedOption('-W'))
        # FIXME: Weird. This option isn't really separate, --param=a=b
        # works. An alias somewhere?
        self.addOption(SeparateOption('--param'))

        # FIXME: What is this? Seems to do something on Linux. I think
        # only one is valid, but have a log that uses both.
        self.addOption(FlagOption('-pthread'))
        self.addOption(FlagOption('-pthreads'))

    def addOption(self, opt):
        self.options.append(opt)
        return opt

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
                args.append(PositionalArg(i, self.inputOption))
        return args
    
    def lookupOptForArg(self, i, string, it):
        for o in self.options:
            arg = o.accept(i, string, it)
            if arg is not None:
                return arg
        return PositionalArg(i, self.unknownOption)
