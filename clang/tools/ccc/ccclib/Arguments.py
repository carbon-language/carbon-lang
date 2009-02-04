
###

class InvalidArgumentsError(ValueError):
    """InvalidArgumentsError - The driver arguments are invalid or
    inconsistent."""

class MissingArgumentError(ValueError):
    """MissingArgumentError - An option required an argument but none
    was given."""

###

class Option(object):
    """Option - Root option class."""

    def __init__(self, name, group=None, alias=None,
                 isLinkerInput=False, noOptAsInput=False,
                 forceSeparateRender=False,
                 forceJoinedRender=False,
                 unsupported=False):
        assert group is None or isinstance(group, OptionGroup)
        # Multi-level aliases are not supported, and alias options
        # cannot have groups. This just simplifies option tracking, it
        # is not an inherent limitation.
        assert alias is None or (alias.alias is None and
                                 group is None)
        
        self.name = name
        self.group = group
        self.alias = alias
        self.isLinkerInput = isLinkerInput
        self.noOptAsInput = noOptAsInput
        self.forceSeparateRender = forceSeparateRender
        self.forceJoinedRender = forceJoinedRender
        self.unsupported = unsupported

    def getUnaliasedOption(self):
        if self.alias:
            return self.alias.getUnaliasedOption()
        return self

    def getRenderName(self):
        return self.getUnaliasedOption().name

    def matches(self, opt):
        """matches(opt) -> bool
        
        Predicate for whether this option is part of the given option
        (which may be a group)."""

        if self.alias:
            return self.alias.matches(opt)
        if self is opt:
            return True
        elif self.group:
            return self.group.matches(opt)
        else:
            return False

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

    def forwardToGCC(self):
        # FIXME: Get rid of this hack.
        if self.name == '<input>':
            return False

        if self.isLinkerInput:
            return False

        return self.name not in  ('-E', '-S', '-c',
                                  '-arch', '-fsyntax-only', '-combine', '-x',
                                  '-###', '-o')

class OptionGroup(Option):
    """OptionGroup - A fake option class used to group options so that
    the driver can efficiently refer to an entire set of options."""

    def __init__(self, name, group=None):
        super(OptionGroup, self).__init__(name, group)

    def accept(self, index, arg, it):
        raise RuntimeError,"accept() should never be called on an OptionGroup"
        
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

class CommaJoinedOption(Option):
    """An option which literally prefixs its argument, but which
    conceptually may have an arbitrary number of arguments which are
    separated by commas."""

    def accept(self, index, arg, it):
        if arg.startswith(self.name):
            return CommaJoinedValuesArg(index, self)

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
        if arg == self.name:
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
        return [self.opt.getRenderName()]

    def renderAsInput(self, args):
        return self.render(args)

class ValueArg(Arg):
    """ValueArg - An instance of an option which has an argument."""

    def getValue(self, args):
        abstract

    def getValues(self, args):
        return [self.getValue(args)]

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
        if self.opt.forceSeparateRender:
            return [self.opt.getRenderName(),
                    self.getValue(args)]
        return [self.opt.getRenderName() + self.getValue(args)]

    def renderAsInput(self, args):
        if self.opt.noOptAsInput:
            return [self.getValue(args)]
        return self.render(args)

class SeparateValueArg(ValueArg):
    """SeparateValueArg - A single value argument where the value
    follows the option in the argument vector."""

    def getValue(self, args):
        return args.getInputString(self.index, offset=1)

    def render(self, args):
        if self.opt.forceJoinedRender:
            return [self.opt.getRenderName() + self.getValue(args)]
        return [self.opt.getRenderName(), self.getValue(args)]

    def renderAsInput(self, args):
        if self.opt.noOptAsInput:
            return [self.getValue(args)]
        return self.render(args)

class MultipleValuesArg(Arg):
    """MultipleValuesArg - An argument with multiple values which
    follow the option in the argument vector."""

    # FIXME: Should we unify this with SeparateValueArg?

    def getValues(self, args):
        return [args.getInputString(self.index, offset=1+i)
                for i in range(self.opt.numArgs)]

    def render(self, args):
        return [self.opt.getRenderName()] + self.getValues(args)

class CommaJoinedValuesArg(Arg):
    """CommaJoinedValuesArg - An argument with multiple values joined
    by commas and joined (suffixed) to the option.
    
    The key point of this arg is that it renders its values into
    separate arguments, which allows it to be used as a generic
    mechanism for passing arguments through to tools."""
    
    def getValues(self, args):
        return args.getInputString(self.index)[len(self.opt.name):].split(',')

    def render(self, args):
        return [self.opt.getRenderName() + ','.join(self.getValues(args))]
    
    def renderAsInput(self, args):
        return self.getValues(args)

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
        return ([self.opt.getRenderName() + self.getJoinedValue(args)] + 
                [self.getSeparateValue(args)])

###

class InputIndex:
    def __init__(self, sourceId, pos):
        self.sourceId = sourceId
        self.pos = pos

    def __repr__(self):
        return 'InputIndex(%d, %d)' % (self.sourceId, self.pos)

class ArgList(object):
    """ArgList - Collect an input argument vector along with a set of
    parsed Args and supporting information."""

    def __init__(self, parser, argv):
        self.parser = parser
        self.argv = list(argv)
        self.syntheticArgv = []
        self.lastArgs = {}
        self.args = []

    def getArgs(self, option):
        # FIXME: How efficient do we want to make this. One reasonable
        # solution would be to embed a linked list inside each arg and
        # automatically chain them (with pointers to head and
        # tail). This gives us efficient access to the (first, last,
        # all) arg(s) with little overhead.
        for arg in self.args:
            if arg.opt.matches(option):
                yield arg

    def getArgs2(self, optionA, optionB):
        """getArgs2 - Iterate over all arguments for two options, in
        the order they were specified."""
        # As long as getArgs is efficient, we can easily make this
        # efficient by iterating both at once and always taking the
        # earlier arg.
        for arg in self.args:
            if (arg.opt.matches(optionA) or
                arg.opt.matches(optionB)):
                yield arg

    def getArgs3(self, optionA, optionB, optionC):
        """getArgs3 - Iterate over all arguments for three options, in
        the order they were specified."""
        for arg in self.args:
            if (arg.opt.matches(optionA) or
                arg.opt.matches(optionB) or
                arg.opt.matches(optionC)):
                yield arg

    def getLastArg(self, option):
        return self.lastArgs.get(option)

    def getInputString(self, index, offset=0):
        # Source 0 is argv.
        if index.sourceId == 0:
            return self.argv[index.pos + offset]
        
        # Source 1 is synthetic argv.
        if index.sourceId == 1:
            return self.syntheticArgv[index.pos + offset]

        raise RuntimeError,'Unknown source ID for index.'

    def addLastArg(self, output, option):
        """addLastArgs - Extend the given output vector with the last
        instance of a given option."""
        arg = self.getLastArg(option)
        if arg:
            output.extend(self.render(arg))

    def addAllArgs(self, output, option):
        """addAllArgs - Extend the given output vector with all
        instances of a given option."""
        for arg in self.getArgs(option):
            output.extend(self.render(arg))

    def addAllArgs2(self, output, optionA, optionB):
        """addAllArgs2 - Extend the given output vector with all
        instances of two given option, with relative order preserved."""
        for arg in self.getArgs2(optionA, optionB):
            output.extend(self.render(arg))

    def addAllArgs3(self, output, optionA, optionB, optionC):
        """addAllArgs3 - Extend the given output vector with all
        instances of three given option, with relative order preserved."""
        for arg in self.getArgs3(optionA, optionB, optionC):
            output.extend(self.render(arg))

    def addAllArgsTranslated(self, output, option, translation):
        """addAllArgsTranslated - Extend the given output vector with
        all instances of a given option, rendered as separate
        arguments with the actual option name translated to a user
        specified string. For example, '-foox' will be render as
        ['-bar', 'x'] if '-foo' was the option and '-bar' was the
        translation.
        
        This routine expects that the option can only yield ValueArg
        instances."""
        for arg in self.getArgs(option):
            assert isinstance(arg, ValueArg)
            output.append(translation)
            output.append(self.getValue(arg))

    def makeIndex(self, *strings):
        pos = len(self.syntheticArgv)
        self.syntheticArgv.extend(strings)
        return InputIndex(1, pos)

    def makeFlagArg(self, option):
        return Arg(self.makeIndex(option.name),
                   option)

    def makeInputArg(self, string):
        return PositionalArg(self.makeIndex(string),
                             self.parser.inputOption)

    def makeUnknownArg(self, string):
        return PositionalArg(self.makeIndex(string),
                             self.parser.unknownOption)

    def makeSeparateArg(self, string, option):
        return SeparateValueArg(self.makeIndex(option.name, string),
                                option)

    def makeJoinedArg(self, string, option):
        return JoinedValueArg(self.makeIndex(option.name + string),
                              option)

    # Support use as a simple arg list.

    def __iter__(self):
        return iter(self.args)

    def append(self, arg):
        if arg.opt.unsupported:
            raise InvalidArgumentsError('option %r is unsupported' % arg.opt.name)

        self.args.append(arg)
        
        opt = arg.opt
        if opt.alias:
            opt = opt.alias
        self.lastArgs[opt] = arg
        if opt.group is not None:
            self.lastArgs[opt.group] = arg

    # Forwarding methods.
    #
    # FIXME: Clean this up once restructuring is done.

    def render(self, arg):
        return arg.render(self)

    def renderAsInput(self, arg):
        return arg.renderAsInput(self)

    def getValue(self, arg):
        return arg.getValue(self)

    def getValues(self, arg):
        return arg.getValues(self)

    def getSeparateValue(self, arg):
        return arg.getSeparateValue(self)

    def getJoinedValue(self, arg):
        return arg.getJoinedValue(self)

class DerivedArgList(ArgList):
    def __init__(self, args):
        super(DerivedArgList, self).__init__(args.parser, args.argv)
        self.parser = args.parser
        self.argv = args.argv
        self.syntheticArgv = args.syntheticArgv
        self.lastArgs = {}
        self.args = []
        
###
    
class OptionParser:
    def __init__(self):
        self.options = []
        self.inputOption = InputOption()
        self.unknownOption = UnknownOption()

        # Driver driver options
        self.archOption = self.addOption(SeparateOption('-arch'))

        # Misc driver options
        self.passExitCodesOption = self.addOption(FlagOption('-pass-exit-codes'))
        self.dumpspecsOption = self.addOption(FlagOption('-dumpspecs'))
        self.dumpversionOption = self.addOption(FlagOption('-dumpversion'))
        self.dumpmachineOption = self.addOption(FlagOption('-dumpmachine'))
        self.printSearchDirsOption = self.addOption(FlagOption('-print-search-dirs'))
        self.printLibgccFileNameOption = self.addOption(FlagOption('-print-libgcc-file-name'))
        self.printFileNameOption = self.addOption(JoinedOption('-print-file-name='))
        self.printProgNameOption = self.addOption(JoinedOption('-print-prog-name='))
        self.printMultiDirectoryOption = self.addOption(FlagOption('-print-multi-directory'))
        self.printMultiOsDirectoryOption = self.addOption(FlagOption('-print-multi-os-directory'))
        self.printMultiLibOption = self.addOption(FlagOption('-print-multi-lib'))

        # Pipeline control
        self.hashHashHashOption = self.addOption(FlagOption('-###'))
        self.EOption = self.addOption(FlagOption('-E'))
        self.SOption = self.addOption(FlagOption('-S'))
        self.cOption = self.addOption(FlagOption('-c'))
        self.combineOption = self.addOption(FlagOption('-combine', unsupported=True))
        self.noIntegratedCPPOption = self.addOption(FlagOption('-no-integrated-cpp'))
        self.pipeOption = self.addOption(FlagOption('-pipe'))
        self.saveTempsOption = self.addOption(FlagOption('-save-temps'))

        # FIXME: Error out if this is used.
        self.specsOption = self.addOption(JoinedOption('-specs='))
        # FIXME: Implement.
        self.addOption(FlagOption('-time'))
        # FIXME: Implement.
        self.vOption = self.addOption(FlagOption('-v'))

        # Input/output stuff
        self.oOption = self.addOption(JoinedOrSeparateOption('-o', noOptAsInput=True))
        self.xOption = self.addOption(JoinedOrSeparateOption('-x'))

        self.ObjCOption = self.addOption(FlagOption('-ObjC'))
        self.ObjCXXOption = self.addOption(FlagOption('-ObjC++'))

        # FIXME: Weird, gcc claims this here in help but I'm not sure why;
        # perhaps interaction with preprocessor? Investigate.
        
        # FIXME: This is broken in Darwin cc1, it wants std* and this
        # is std=. May need an option group for this as well.
        self.stdOption = self.addOption(JoinedOption('-std='))
        self.addOption(JoinedOrSeparateOption('--sysroot'))

        # Blanket pass-through options.

        self.XanalyzerOption = self.addOption(SeparateOption('-Xanalyzer'))
        self.XclangOption = self.addOption(SeparateOption('-Xclang'))

        self.WaOption = self.addOption(CommaJoinedOption('-Wa,'))
        self.XassemblerOption = self.addOption(SeparateOption('-Xassembler'))

        self.WpOption = self.addOption(CommaJoinedOption('-Wp,'))
        self.XpreprocessorOption = self.addOption(SeparateOption('-Xpreprocessor'))

        self.addOption(CommaJoinedOption('-Wl,', isLinkerInput=True))
        self.XlinkerOption = self.addOption(SeparateOption('-Xlinker', 
                                                           isLinkerInput=True, 
                                                           noOptAsInput=True))

        ####
        # Bring on the random garbage.

        self.sOption = self.addOption(FlagOption('-s'))

        self.MGroup = OptionGroup('-M')
        self.MOption = self.addOption(FlagOption('-M', self.MGroup))
        self.MDOption = self.addOption(FlagOption('-MD', self.MGroup))
        self.MGOption = self.addOption(FlagOption('-MG', self.MGroup))
        self.MMDOption = self.addOption(FlagOption('-MMD', self.MGroup))
        self.MPOption = self.addOption(FlagOption('-MP', self.MGroup))
        self.MMOption = self.addOption(FlagOption('-MM', self.MGroup))
        self.MFOption = self.addOption(JoinedOrSeparateOption('-MF', self.MGroup))
        self.MTOption = self.addOption(JoinedOrSeparateOption('-MT', self.MGroup))
        self.MQOption = self.addOption(JoinedOrSeparateOption('-MQ', self.MGroup))
        self.MachOption = self.addOption(FlagOption('-Mach'))
        self.uGroup = OptionGroup('-u')
        self.undefOption = self.addOption(FlagOption('-undef', self.uGroup))

        self.wOption = self.addOption(FlagOption('-w'))
        self.bundle_loaderOption = self.addOption(SeparateOption('-bundle_loader'))
        self.bundleOption = self.addOption(FlagOption('-bundle'))
        self.client_nameOption = self.addOption(JoinedOrSeparateOption('-client_name'))
        self.compatibility_versionOption = self.addOption(JoinedOrSeparateOption('-compatibility_version'))
        self.current_versionOption = self.addOption(JoinedOrSeparateOption('-current_version'))
        self.dependencyFileOption = self.addOption(SeparateOption('-dependency-file'))
        self.dynamiclibOption = self.addOption(FlagOption('-dynamiclib'))
        self.dynamicOption = self.addOption(FlagOption('-dynamic'))
        self.dylinker_install_nameOption = self.addOption(JoinedOrSeparateOption('-dylinker_install_name'))
        self.dylinkerOption = self.addOption(FlagOption('-dylinker'))

        self.iGroup = OptionGroup('-i')
        self.idirafterOption = self.addOption(JoinedOrSeparateOption('-idirafter', self.iGroup))
        self.imacrosOption = self.addOption(JoinedOrSeparateOption('-imacros', self.iGroup))
        self.iprefixOption = self.addOption(JoinedOrSeparateOption('-iprefix', self.iGroup))
        self.iwithprefixOption = self.addOption(JoinedOrSeparateOption('-iwithprefix', self.iGroup))
        self.iwithprefixbeforeOption = self.addOption(JoinedOrSeparateOption('-iwithprefixbefore', self.iGroup))
        self.addOption(JoinedOrSeparateOption('-iquote', self.iGroup))
        self.isysrootOption = self.addOption(JoinedOrSeparateOption('-isysroot', self.iGroup))
        self.includeOption = self.addOption(JoinedOrSeparateOption('-include', self.iGroup))

        self.keep_private_externsOption = self.addOption(JoinedOrSeparateOption('-keep_private_externs'))
        self.private_bundleOption = self.addOption(FlagOption('-private_bundle'))
        self.seg1addrOption = self.addOption(JoinedOrSeparateOption('-seg1addr'))
        self.segprotOption = self.addOption(JoinedOrSeparateOption('-segprot'))
        self.sub_libraryOption = self.addOption(JoinedOrSeparateOption('-sub_library'))
        self.sub_umbrellaOption = self.addOption(JoinedOrSeparateOption('-sub_umbrella'))
        self.umbrellaOption = self.addOption(SeparateOption('-umbrella', self.uGroup))
        self.undefinedOption = self.addOption(JoinedOrSeparateOption('-undefined', self.uGroup))
        self.headerpad_max_install_namesOption = self.addOption(JoinedOption('-headerpad_max_install_names'))
        self.twolevel_namespaceOption = self.addOption(FlagOption('-twolevel_namespace'))
        self.twolevel_namespace_hintsOption = self.addOption(FlagOption('-twolevel_namespace_hints'))
        self.prebindOption = self.addOption(FlagOption('-prebind'))
        self.noprebindOption = self.addOption(FlagOption('-noprebind'))
        self.nofixprebindingOption = self.addOption(FlagOption('-nofixprebinding'))
        self.prebind_all_twolevel_modulesOption = self.addOption(FlagOption('-prebind_all_twolevel_modules'))
        self.remapOption = self.addOption(FlagOption('-remap'))
        self.read_only_relocsOption = self.addOption(SeparateOption('-read_only_relocs'))
        self.nomultidefsOption = self.addOption(FlagOption('-nomultidefs'))
        self.nostartfilesOption = self.addOption(FlagOption('-nostartfiles'))
        self.nodefaultlibsOption = self.addOption(FlagOption('-nodefaultlibs'))
        self.nostdlibOption = self.addOption(FlagOption('-nostdlib'))
        self.nostdincOption = self.addOption(FlagOption('-nostdinc'))
        self.objectOption = self.addOption(FlagOption('-object'))
        self.preloadOption = self.addOption(FlagOption('-preload'))
        self.staticOption = self.addOption(FlagOption('-static'))
        self.pagezero_sizeOption = self.addOption(FlagOption('-pagezero_size'))
        self.sharedOption = self.addOption(FlagOption('-shared'))
        self.staticLibgccOption = self.addOption(FlagOption('-static-libgcc'))
        self.sharedLibgccOption = self.addOption(FlagOption('-shared-libgcc'))
        self.COption = self.addOption(FlagOption('-C'))
        self.CCOption = self.addOption(FlagOption('-CC'))
        self.HOption = self.addOption(FlagOption('-H'))
        self.addOption(FlagOption('-R'))
        self.POption = self.addOption(FlagOption('-P'))
        self.QOption = self.addOption(FlagOption('-Q'))
        self.QnOption = self.addOption(FlagOption('-Qn'))
        self.addOption(FlagOption('--constant-cfstrings'))
        self.traditionalOption = self.addOption(FlagOption('-traditional'))
        self.traditionalCPPOption = self.addOption(FlagOption('-traditional-cpp'))
        # FIXME: Alias.
        self.addOption(FlagOption('--traditional'))
        self.no_dead_strip_inits_and_termsOption = self.addOption(FlagOption('-no_dead_strip_inits_and_terms'))
        self.addOption(JoinedOption('-weak-l', isLinkerInput=True))
        self.addOption(SeparateOption('-weak_framework', isLinkerInput=True))
        self.addOption(SeparateOption('-weak_library', isLinkerInput=True))
        self.whyloadOption = self.addOption(FlagOption('-whyload'))
        self.whatsloadedOption = self.addOption(FlagOption('-whatsloaded'))
        self.sectalignOption = self.addOption(MultiArgOption('-sectalign', numArgs=3))
        self.sectobjectsymbolsOption = self.addOption(MultiArgOption('-sectobjectsymbols', numArgs=2))
        self.segcreateOption = self.addOption(MultiArgOption('-segcreate', numArgs=3))
        self.seglinkeditOption = self.addOption(FlagOption('-seglinkedit'))
        self.noseglinkeditOption = self.addOption(FlagOption('-noseglinkedit'))
        self.sectcreateOption = self.addOption(MultiArgOption('-sectcreate', numArgs=3))
        self.sectorderOption = self.addOption(MultiArgOption('-sectorder', numArgs=3))

        self.all_loadOption = self.addOption(FlagOption('-all_load'))
        self.allowable_clientOption = self.addOption(SeparateOption('-allowable_client'))
        self.bind_at_loadOption = self.addOption(FlagOption('-bind_at_load'))
        self.dead_stripOption = self.addOption(FlagOption('-dead_strip'))
        self.dylib_fileOption = self.addOption(SeparateOption('-dylib_file'))
        self.exported_symbols_listOption = self.addOption(SeparateOption('-exported_symbols_list'))
        self.flat_namespaceOption = self.addOption(FlagOption('-flat_namespace'))
        self.force_cpusubtype_ALLOption = self.addOption(FlagOption('-force_cpusubtype_ALL'))
        self.force_flat_namespaceOption = self.addOption(FlagOption('-force_flat_namespace'))
        self.image_baseOption = self.addOption(FlagOption('-image_base'))
        self.initOption = self.addOption(SeparateOption('-init'))
        self.install_nameOption = self.addOption(SeparateOption('-install_name'))
        self.multi_moduleOption = self.addOption(FlagOption('-multi_module'))
        self.multiply_definedOption = self.addOption(SeparateOption('-multiply_defined'))
        self.multiply_defined_unusedOption = self.addOption(SeparateOption('-multiply_defined_unused'))
        self.seg_addr_table_filenameOption = self.addOption(SeparateOption('-seg_addr_table_filename'))
        self.seg_addr_tableOption = self.addOption(SeparateOption('-seg_addr_table'))
        self.segaddrOption = self.addOption(SeparateOption('-segaddr'))
        self.segs_read_only_addrOption = self.addOption(SeparateOption('-segs_read_only_addr'))
        self.segs_read_write_addrOption = self.addOption(SeparateOption('-segs_read_write_addr'))
        # FIXME: This probably isn't necessary.
        self.segs_read_Option = self.addOption(JoinedOption('-segs_read_'))
        self.single_moduleOption = self.addOption(FlagOption('-single_module'))
        self.unexported_symbols_listOption = self.addOption(SeparateOption('-unexported_symbols_list', self.uGroup))
        self.weak_reference_mismatchesOption = self.addOption(SeparateOption('-weak_reference_mismatches'))

        self.addOption(SeparateOption('-filelist', isLinkerInput=True))
        self.addOption(SeparateOption('-framework', isLinkerInput=True))

        self.addOption(JoinedOption('-i', self.iGroup))

        self.emitLLVMOption = self.addOption(FlagOption('-emit-llvm'))
        self.eOption = self.addOption(JoinedOrSeparateOption('-e'))
        self.rOption = self.addOption(JoinedOrSeparateOption('-r'))

        self.pgOption = self.addOption(FlagOption('-pg'))
        self.pOption = self.addOption(FlagOption('-p'))

        doNotReallySupport = 1
        if doNotReallySupport:
            # Archaic gcc option.
            self.addOption(FlagOption('-cpp-precomp'))
            self.addOption(FlagOption('-no-cpp-precomp'))

        # C options for testing

        self.trigraphsOption = self.addOption(FlagOption('-trigraphs'))

        # FIXME: This is broken, we need -A as a single option to send
        # stuff to cc1, but the way the ld spec is constructed it
        # wants to see -A options but only as a separate arg.
        self.AOption = self.addOption(JoinedOrSeparateOption('-A'))
        self.DOption = self.addOption(JoinedOrSeparateOption('-D'))
        self.FOption = self.addOption(JoinedOrSeparateOption('-F'))

        self.IGroup = OptionGroup('-I')
        self.I_Option = self.addOption(FlagOption('-I-', self.IGroup))
        self.IOption = self.addOption(JoinedOrSeparateOption('-I', self.IGroup))

        self.LOption = self.addOption(JoinedOrSeparateOption('-L'))
        self.TOption = self.addOption(JoinedOrSeparateOption('-T'))
        self.UOption = self.addOption(JoinedOrSeparateOption('-U'))
        self.ZOption = self.addOption(JoinedOrSeparateOption('-Z'))

        self.addOption(JoinedOrSeparateOption('-l', isLinkerInput=True))
        self.uOption = self.addOption(JoinedOrSeparateOption('-u', self.uGroup))
        self.tOption = self.addOption(JoinedOrSeparateOption('-t'))
        self.yOption = self.addOption(JoinedOption('-y'))

        # FIXME: What is going on here? '-X' goes to linker, and -X ... goes nowhere?
        self.XOption = self.addOption(FlagOption('-X'))
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
        # FIXME: Naming convention.
        self.dGroup = OptionGroup('-d')
        self.dAOption = self.addOption(FlagOption('-dA', self.dGroup))
        self.dOption = self.addOption(FlagOption('-d', self.dGroup))

        # Use a group for this in anticipation of adding more -d
        # options explicitly. Note that we don't put many -d things in
        # the -d group (like -dylinker, or '-d' by itself) because it
        # is really a gcc bug that it ships these to cc1.
        self.dGroup = OptionGroup('-d')
        self.addOption(JoinedOption('-d', group=self.dGroup))

        self.gGroup = OptionGroup('-g')
        self.gfullOption = self.addOption(JoinedOption('-gfull', self.gGroup))
        self.gusedOption = self.addOption(JoinedOption('-gused', self.gGroup))
        self.gstabsOption = self.addOption(JoinedOption('-gstabs', self.gGroup))
        self.g0Option = self.addOption(JoinedOption('-g0', self.gGroup))
        self.g3Option = self.addOption(JoinedOption('-g3', self.gGroup))
        # FIXME: Naming.
        self.gOption = self.addOption(FlagOption('-g', self.gGroup))
        self.addOption(JoinedOption('-g', self.gGroup))

        # FIXME: How should we handle clang specific options? Do we
        # want to avoid passing them to gcc/cc1 (which will generally
        # not eat them), or should we let the user sort it out.

        self.fblocksGroup = OptionGroup('-fblocks')
        self.f_blocks = self.addOption(FlagOption('-fblocks', self.fblocksGroup))
        self.f_noblocks = self.addOption(FlagOption('-fno-blocks', self.fblocksGroup))
        # self.fblocksOption = self.addOption(JoinedOption('-fblocks', self.fblocksGroup))
        
        self.fGroup = OptionGroup('-f')
        self.Clang_fGroup = OptionGroup('-f', self.fGroup)
        self.fastOption = self.addOption(FlagOption('-fast', self.fGroup))
        self.fastfOption = self.addOption(FlagOption('-fastf', self.fGroup))
        self.fastcpOption = self.addOption(FlagOption('-fastcp', self.fGroup))

        self.f_appleKextOption = self.addOption(FlagOption('-fapple-kext', self.fGroup))
        self.f_bootclasspathOption = self.addOption(JoinedOption('-fbootclasspath=', self.fGroup))        
        self.f_classpathOption = self.addOption(JoinedOption('-fclasspath=', self.fGroup))        
        self.f_compileResourceOption = self.addOption(JoinedOption('-fcompile-resource=', self.fGroup))
        self.f_constantCfstringsOption = self.addOption(FlagOption('-fconstant-cfstrings', self.fGroup))
        self.f_createProfileOption = self.addOption(FlagOption('-fcreate-profile', self.fGroup))
        self.f_debugPassArgumentsOption = self.addOption(FlagOption('-fdebug-pass-arguments', self.fGroup))
        self.f_debugPassStructureOption = self.addOption(FlagOption('-fdebug-pass-structure', self.fGroup))
        self.f_eliminateUnusedDebugSymbolsOption = self.addOption(FlagOption('-feliminate-unused-debug-symbols', self.fGroup))
        self.addOption(FlagOption('-femit-all-decls', self.Clang_fGroup))
        self.f_encodingOption = self.addOption(JoinedOption('-fencoding=', self.fGroup))
        self.f_exceptionsOption = self.addOption(FlagOption('-fexceptions', self.Clang_fGroup))
        self.f_extdirsOption = self.addOption(JoinedOption('-fextdirs=', self.fGroup))
        self.f_gnuRuntimeOption = self.addOption(FlagOption('-fgnu-runtime', self.Clang_fGroup))
        self.f_indirectVirtualCallsOption = self.addOption(FlagOption('-findirect-virtual-calls', self.fGroup))
        self.f_laxVectorConversionsOption = self.addOption(FlagOption('-flax-vector-conversions', self.Clang_fGroup))
        self.f_limitedPrecisionOption = self.addOption(JoinedOption('-flimited-precision=', self.fGroup))
        self.f_msExtensionsOption = self.addOption(FlagOption('-fms-extensions', self.Clang_fGroup))
        self.f_mudflapOption = self.addOption(FlagOption('-fmudflap', self.fGroup))
        self.f_mudflapthOption = self.addOption(FlagOption('-fmudflapth', self.fGroup))
        self.f_nestedFunctionsOption = self.addOption(FlagOption('-fnested-functions', self.fGroup))
        self.f_nextRuntimeOption = self.addOption(FlagOption('-fnext-runtime', self.Clang_fGroup))
        self.f_noCaretDiagnosticsOption = self.addOption(FlagOption('-fno-caret-diagnostics', self.Clang_fGroup))
        self.f_noConstantCfstringsOption = self.addOption(FlagOption('-fno-constant-cfstrings', self.fGroup))
        self.f_noEliminateUnusedDebugSymbolsOption = self.addOption(FlagOption('-fno-eliminate-unused-debug-symbols', self.fGroup))
        self.f_noPascalStringsOption = self.addOption(FlagOption('-fno-pascal-strings', self.fGroup))
        self.f_noShowColumnOption = self.addOption(FlagOption('-fno-show-column', self.Clang_fGroup))
        self.f_noWorkingDirectoryOption = self.addOption(FlagOption('-fno-working-directory', self.fGroup))
        self.f_objcGcOnlyOption = self.addOption(FlagOption('-fobjc-gc-only', self.Clang_fGroup))
        self.f_objcGcOption = self.addOption(FlagOption('-fobjc-gc', self.Clang_fGroup))
        self.f_objcNonfragileAbiOption = self.addOption(FlagOption('-fobjc-nonfragile-abi', self.Clang_fGroup))
        self.f_objcOption = self.addOption(FlagOption('-fobjc', self.fGroup))
        self.f_omitFramePointerOption = self.addOption(FlagOption('-fomit-frame-pointer', self.fGroup))
        self.f_openmpOption = self.addOption(FlagOption('-fopenmp', self.fGroup))
        self.f_outputClassDirOption = self.addOption(JoinedOption('-foutput-class-dir=', self.fGroup))
        self.f_pascalStringsOption = self.addOption(FlagOption('-fpascal-strings', self.Clang_fGroup))
        self.f_pieOption = self.addOption(FlagOption('-fpie', self.fGroup))
        self.f_PIEOption = self.addOption(FlagOption('-fPIE', self.fGroup))
        self.f_picOption = self.addOption(FlagOption('-fpic', self.fGroup))
        self.f_PICOption = self.addOption(FlagOption('-fPIC', self.fGroup))
        self.f_profileArcsOption = self.addOption(FlagOption('-fprofile-arcs', self.fGroup))
        self.f_profileGenerateOption = self.addOption(FlagOption('-fprofile-generate', self.fGroup))
        self.f_terminatedVtablesOption = self.addOption(FlagOption('-fterminated-vtables', self.fGroup))
        self.f_timeReportOption = self.addOption(FlagOption('-ftime-report', self.fGroup))
        self.f_traditionalOption = self.addOption(FlagOption('-ftraditional', self.fGroup))
        self.f_unwindTablesOption = self.addOption(FlagOption('-funwind-tables', self.fGroup))
        self.f_writableStringsOption = self.addOption(FlagOption('-fwritable-strings', self.Clang_fGroup))
        self.f_zeroInitializedInBssOption = self.addOption(FlagOption('-fzero-initialized-in-bss', self.fGroup))
        self.fOption = self.addOption(JoinedOption('-f', self.fGroup))

        self.coverageOption = self.addOption(FlagOption('-coverage'))

        self.mGroup = OptionGroup('-m')
        self.m_32Option = self.addOption(FlagOption('-m32', self.mGroup))
        self.m_64Option = self.addOption(FlagOption('-m64', self.mGroup))
        self.m_dynamicNoPicOption = self.addOption(JoinedOption('-mdynamic-no-pic', self.mGroup))
        self.m_iphoneosVersionMinOption = self.addOption(JoinedOption('-miphoneos-version-min=', self.mGroup))
        self.m_kernelOption = self.addOption(FlagOption('-mkernel', self.mGroup))
        self.m_macosxVersionMinOption = self.addOption(JoinedOption('-mmacosx-version-min=', self.mGroup))
        self.m_constantCfstringsOption = self.addOption(FlagOption('-mconstant-cfstrings', self.mGroup))
        self.m_noConstantCfstringsOption = self.addOption(FlagOption('-mno-constant-cfstrings', self.mGroup))
        self.m_warnNonportableCfstringsOption = self.addOption(FlagOption('-mwarn-nonportable-cfstrings', self.mGroup))
        self.m_noWarnNonportableCfstringsOption = self.addOption(FlagOption('-mno-warn-nonportable-cfstrings', self.mGroup))
        self.m_pascalStringsOption = self.addOption(FlagOption('-mpascal-strings', self.mGroup))
        self.m_noPascalStringsOption = self.addOption(FlagOption('-mno-pascal-strings', self.mGroup))
        self.m_tuneOption = self.addOption(JoinedOption('-mtune=', self.mGroup))

        # Ugh. Need to disambiguate our naming convetion. -m x goes to
        # the linker sometimes, wheres -mxxxx is used for a variety of
        # other things.        
        self.mSeparate = self.addOption(SeparateOption('-m', self.mGroup))
        self.mJoined = self.addOption(JoinedOption('-m', self.mGroup))

        # FIXME: Why does Darwin send -a* to cc1?
        self.aGroup = OptionGroup('-a')
        self.ansiOption = self.addOption(FlagOption('-ansi', self.aGroup))

        self.aOption = self.addOption(JoinedOption('-a', self.aGroup))

        self.pedanticGroup = OptionGroup('-pedantic')
        self.pedanticOption = self.addOption(FlagOption('-pedantic', self.pedanticGroup))
        self.pedanticErrorsOption = self.addOption(FlagOption('-pedantic-errors', self.pedanticGroup))
        self.OOption = self.addOption(JoinedOption('-O'))

        self.WGroup = OptionGroup('-W')
        self.ClangWGroup = OptionGroup('-W', self.WGroup)

        self.WallOption = self.addOption(FlagOption('-Wall', self.WGroup))
        self.addOption(FlagOption('-Wunused-macros', self.ClangWGroup))
        self.addOption(FlagOption('-Wfloat-equal', self.ClangWGroup))
        self.addOption(FlagOption('-Wreadonly-setter-attrs', self.ClangWGroup))
        self.addOption(FlagOption('-Wno-format-nonliteral', self.ClangWGroup))
        self.addOption(FlagOption('-Wundef', self.ClangWGroup))
        self.addOption(FlagOption('-Wimplicit-function-declaration', self.ClangWGroup))
        self.addOption(FlagOption('-Wno-strict-selector-match', self.ClangWGroup))

        self.WnonportableCfstringsOption = self.addOption(JoinedOption('-Wnonportable-cfstrings', self.WGroup))
        self.WnoNonportableCfstringsOption = self.addOption(JoinedOption('-Wno-nonportable-cfstrings', self.WGroup))
        self.WOption = self.addOption(JoinedOption('-W', self.WGroup))

        # FIXME: Weird. This option isn't really separate, --param=a=b
        # works. There is something else going on which interprets the
        # '='.
        self._paramOption = self.addOption(SeparateOption('--param'))

        # FIXME: What is this? I think only one is valid, but have a
        # log that uses both.
        self.pthreadOption = self.addOption(FlagOption('-pthread'))
        self.addOption(FlagOption('-pthreads'))

        # Version control.
        # FIXME: Figure out what to do about these.
        self.BOption = self.addOption(JoinedOrSeparateOption('-B', unsupported=True))
        self.addOption(JoinedOrSeparateOption('-V', unsupported=True))
        self.addOption(JoinedOrSeparateOption('-b', unsupported=True))

        # Clang static analyzer options (also see -WA,).
        self.analyzeOption = self.addOption(FlagOption('--analyze'))

        # Long option handling (aliases).
        self.addOption(FlagOption('--help'))
        self.addOption(FlagOption('--target-help'))
        # FIXME: Who handles this?
        self.addOption(FlagOption('--version'))

        self.addOption(FlagOption('--all-warnings', alias=self.WallOption))
        self.addOption(FlagOption('--ansi', alias=self.ansiOption))
        self.addOption(FlagOption('--assemble', alias=self.SOption))
        self.addOption(FlagOption('--combine', alias=self.combineOption))
        self.addOption(FlagOption('--comments', alias=self.COption))
        self.addOption(FlagOption('--comments-in-macros', alias=self.CCOption))
        self.addOption(FlagOption('--compile', alias=self.cOption))
        self.addOption(FlagOption('--coverage', alias=self.coverageOption))
        self.addOption(FlagOption('--dependencies', alias=self.MOption))
        self.addOption(FlagOption('--entry', alias=self.eOption))
        self.addOption(FlagOption('--extra-warnings', alias=self.WOption))
        self.addOption(FlagOption('--coverage', alias=self.coverageOption))
        self.addOption(FlagOption('--include-barrier', alias=self.I_Option))
        self.addOption(FlagOption('--no-integrated-cpp', alias=self.noIntegratedCPPOption))
        self.addOption(FlagOption('--no-line-commands', alias=self.POption))
#        self.addOption(FlagOption('--no-precompiled-includes', alias=self.noprecompOption))
        self.addOption(FlagOption('--no-standard-includes', alias=self.nostdincOption))
        self.addOption(FlagOption('--no-standard-libraries', alias=self.nostdlibOption))
        self.addOption(FlagOption('--no-warnings', alias=self.wOption))
        self.addOption(FlagOption('--pass-exit-codes', alias=self.passExitCodesOption))
        self.addOption(FlagOption('--pedantic', alias=self.pedanticOption))
        self.addOption(FlagOption('--pedantic-errors', alias=self.pedanticErrorsOption))
#        self.addOption(FlagOption('--pie', alias=self.pieOption))
        self.addOption(FlagOption('--pipe', alias=self.pipeOption))
        self.addOption(FlagOption('--preprocess', alias=self.EOption))
        self.addOption(JoinedOption('--param=', alias=self._paramOption,
                                    forceSeparateRender=True))
        self.addOption(JoinedOption('--print-file-name=', alias=self.printFileNameOption))
        self.addOption(SeparateOption('--print-file-name', alias=self.printFileNameOption))
        self.addOption(FlagOption('--print-libgcc-file-name', 
                                  alias=self.printLibgccFileNameOption))
        self.addOption(FlagOption('--print-missing-file-dependencies', alias=self.MGOption))
        self.addOption(FlagOption('--print-multi-lib', alias=self.printMultiLibOption))
        self.addOption(FlagOption('--print-multi-directory', alias=self.printMultiDirectoryOption))
        self.addOption(FlagOption('--print-multi-os-directory', alias=self.printMultiOsDirectoryOption))
        self.addOption(JoinedOption('--print-prog-name=', alias=self.printProgNameOption))
        self.addOption(SeparateOption('--print-prog-name', alias=self.printProgNameOption))
        self.addOption(FlagOption('--print-search-dirs', alias=self.printSearchDirsOption))
        self.addOption(FlagOption('--profile', alias=self.pOption))
        self.addOption(FlagOption('--profile-blocks', alias=self.aOption))
#        self.addOption(FlagOption('--quiet', alias=self.qOption))
#        self.addOption(FlagOption('--shared', alias=self.sharedOption))
#        self.addOption(FlagOption('--silent', alias=self.qOption))
        self.addOption(FlagOption('--static', alias=self.staticOption))
#        self.addOption(FlagOption('--symbolic', alias=self.symbolicOption))
#        self.addOption(FlagOption('--time', alias=self.timeOption))
        self.addOption(FlagOption('--trace-includes', alias=self.HOption))
        self.addOption(FlagOption('--traditional', alias=self.traditionalOption))
        self.addOption(FlagOption('--traditional-cpp', alias=self.traditionalCPPOption))
        self.addOption(FlagOption('--trigraphs', alias=self.trigraphsOption))
        self.addOption(FlagOption('--user-dependencies', alias=self.MMOption))
        self.addOption(FlagOption('--verbose', alias=self.vOption))
        self.addOption(FlagOption('--save-temps', alias=self.saveTempsOption))
        self.addOption(FlagOption('--write-dependencies', alias=self.MDOption))
        self.addOption(FlagOption('--write-user-dependencies', alias=self.MMDOption))

        # Long options with joined & separate forms (coercing to
        # joined form).

        self.addOption(SeparateOption('--assert', alias=self.AOption))
        self.addOption(JoinedOption('--assert=', alias=self.AOption, 
                                    forceSeparateRender=True))
        self.addOption(JoinedOption('--bootclasspath=', alias=self.f_bootclasspathOption))
        self.addOption(SeparateOption('--bootclasspath', alias=self.f_bootclasspathOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--CLASSPATH=', alias=self.f_classpathOption))
        self.addOption(SeparateOption('--CLASSPATH', alias=self.f_classpathOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--classpath=', alias=self.f_classpathOption))
        self.addOption(SeparateOption('--classpath', alias=self.f_classpathOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--define-macro=', alias=self.DOption))
        self.addOption(SeparateOption('--define-macro', alias=self.DOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--encoding=', alias=self.f_encodingOption))
        self.addOption(SeparateOption('--encoding', alias=self.f_encodingOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--extdirs=', alias=self.f_extdirsOption))
        self.addOption(SeparateOption('--extdirs', alias=self.f_extdirsOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--include-directory=', alias=self.IOption))
        self.addOption(SeparateOption('--include-directory', alias=self.IOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--machine=', alias=self.mJoined))
        self.addOption(SeparateOption('--machine', alias=self.mJoined,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--output-class-directory=', alias=self.f_outputClassDirOption))
        self.addOption(SeparateOption('--output-class-directory', alias=self.f_outputClassDirOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--resource=', alias=self.f_compileResourceOption))
        self.addOption(SeparateOption('--resource', alias=self.f_compileResourceOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--specs=', alias=self.specsOption))
        self.addOption(SeparateOption('--specs', alias=self.specsOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--std=', alias=self.stdOption))
        self.addOption(SeparateOption('--std', alias=self.stdOption,
                                      forceJoinedRender=True))
        self.sysrootOption = self.addOption(JoinedOption('--sysroot='))
        self.addOption(SeparateOption('--sysroot', alias=self.sysrootOption,
                                      forceJoinedRender=True))
        self.addOption(JoinedOption('--undefine-macro=', alias=self.UOption))
        self.addOption(SeparateOption('--undefine-macro', alias=self.UOption,
                                    forceJoinedRender=True))

        # Long options with joined & separate forms (coercing to
        # separate form).

#        self.addOption(JoinedOption('--dump=', alias=self.dOption,
#                                    forceSeparateRender=True))
#        self.addOption(SeparateOption('--dump', alias=self.dOption))
#        self.addOption(JoinedOption('--dumpbase=', alias=self.dumpbaseOption,
#                                    forceSeparateRender=True))
#        self.addOption(SeparateOption('--dumpbase', alias=self.dumpbaseOption))
#        self.addOption(JoinedOption('--for-assembler=', alias=self.WaOption,
#                                    forceSeparateRender=True))
#        self.addOption(SeparateOption('--for-assembler', alias=self.WaOption))
        self.addOption(JoinedOption('--for-linker=', alias=self.XlinkerOption,
                                    forceSeparateRender=True,
                                    isLinkerInput=True, 
                                    noOptAsInput=True))
        self.addOption(SeparateOption('--for-linker', alias=self.XlinkerOption,
                                    isLinkerInput=True, 
                                    noOptAsInput=True))
        self.addOption(JoinedOption('--force-link=', alias=self.uOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--force-link', alias=self.uOption))
        self.addOption(JoinedOption('--imacros=', alias=self.imacrosOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--imacros', alias=self.imacrosOption))
        self.addOption(JoinedOption('--include=', alias=self.includeOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--include', alias=self.includeOption))
        self.addOption(JoinedOption('--include-directory-after=', alias=self.idirafterOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--include-directory-after', alias=self.idirafterOption))
        self.addOption(JoinedOption('--include-prefix=', alias=self.iprefixOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--include-prefix', alias=self.iprefixOption))
        self.addOption(JoinedOption('--include-with-prefix=', alias=self.iwithprefixOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--include-with-prefix', alias=self.iwithprefixOption))
        self.addOption(JoinedOption('--include-with-prefix-before=', alias=self.iwithprefixbeforeOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--include-with-prefix-before', alias=self.iwithprefixbeforeOption))
        self.addOption(JoinedOption('--include-with-prefix-after=', alias=self.iwithprefixOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--include-with-prefix-after', alias=self.iwithprefixOption))
        self.addOption(JoinedOption('--language=', alias=self.xOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--language', alias=self.xOption))
        self.addOption(JoinedOption('--library-directory=', alias=self.LOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--library-directory', alias=self.LOption))
        self.addOption(JoinedOption('--output=', alias=self.oOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--output', alias=self.oOption))
        self.addOption(JoinedOption('--prefix=', alias=self.BOption,
                                    forceSeparateRender=True))
        self.addOption(SeparateOption('--prefix', alias=self.BOption))

        # Long options with joined forms. gcc's handling of '=' for
        # long forms makes these a bit odd.
        #
        # FIXME: We do not currently support these options. The
        # problem is that they need to be reparsed in their translated
        # form; they need to map to the correct option and we have to
        # find a way to do so without replicating all the declared
        # names.
        self.addOption(JoinedOption('--debug=', alias=self.gOption,
                                    unsupported=True))
        self.addOption(FlagOption('--debug', alias=self.gOption,
                                    unsupported=True))
        self.addOption(JoinedOption('--machine-=', alias=self.mJoined,
                                    unsupported=True))
        self.addOption(JoinedOption('--machine-', alias=self.mJoined, 
                                    unsupported=True))
        self.addOption(JoinedOption('--optimize=', alias=self.OOption,
                                    unsupported=True))
        self.addOption(FlagOption('--optimize', alias=self.OOption,
                                    unsupported=True))
        self.addOption(JoinedOption('--warn-=', alias=self.WOption,
                                    unsupported=True))
        self.addOption(JoinedOption('--warn-', alias=self.WOption,
                                    unsupported=True))

        # Ugh.
        self.addOption(JoinedOption('--', alias=self.fOption,
                                    unsupported=True))

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
        args = ArgList(self, argv)
        for pos,a in it:
            i = InputIndex(0, pos)
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
