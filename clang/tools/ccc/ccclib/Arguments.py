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

###

class InputIndex:
    def __init__(self, sourceId, pos):
        self.sourceId = sourceId
        self.pos = pos

    def __repr__(self):
        return 'InputIndex(%d, %d)' % (self.sourceId, self.pos)

class ArgList:
    """ArgList - Collect an input argument vector along with a set of parsed Args
    and supporting information."""

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
            if arg.opt is option:
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
        # FIXME: Error out if this is used.
        self.addOption(JoinedOption('-specs='))
        # FIXME: Implement.
        self.addOption(FlagOption('-time'))
        # FIXME: Implement.
        self.addOption(FlagOption('-v'))

        # Input/output stuff
        self.oOption = self.addOption(JoinedOrSeparateOption('-o'))
        self.xOption = self.addOption(JoinedOrSeparateOption('-x'))

        self.ObjCOption = self.addOption(FlagOption('-ObjC'))
        self.ObjCXXOption = self.addOption(FlagOption('-ObjC++'))

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
        self.MachOption = self.addOption(FlagOption('-Mach'))
        self.addOption(FlagOption('-undef'))

        self.wOption = self.addOption(FlagOption('-w'))
        self.addOption(JoinedOrSeparateOption('-allowable_client'))
        self.client_nameOption = self.addOption(JoinedOrSeparateOption('-client_name'))
        self.compatibility_versionOption = self.addOption(JoinedOrSeparateOption('-compatibility_version'))
        self.current_versionOption = self.addOption(JoinedOrSeparateOption('-current_version'))
        self.dylinkerOption = self.addOption(FlagOption('-dylinker'))
        self.dylinker_install_nameOption = self.addOption(JoinedOrSeparateOption('-dylinker_install_name'))
        self.addOption(JoinedOrSeparateOption('-exported_symbols_list'))
        self.addOption(JoinedOrSeparateOption('-idirafter'))
        self.addOption(JoinedOrSeparateOption('-iquote'))
        self.isysrootOption = self.addOption(JoinedOrSeparateOption('-isysroot'))
        self.keep_private_externsOption = self.addOption(JoinedOrSeparateOption('-keep_private_externs'))
        self.private_bundleOption = self.addOption(FlagOption('-private_bundle'))
        self.seg1addrOption = self.addOption(JoinedOrSeparateOption('-seg1addr'))
        self.segprotOption = self.addOption(JoinedOrSeparateOption('-segprot'))
        self.sub_libraryOption = self.addOption(JoinedOrSeparateOption('-sub_library'))
        self.sub_umbrellaOption = self.addOption(JoinedOrSeparateOption('-sub_umbrella'))
        self.umbrellaOption = self.addOption(JoinedOrSeparateOption('-umbrella'))
        self.undefinedOption = self.addOption(JoinedOrSeparateOption('-undefined'))
        self.addOption(JoinedOrSeparateOption('-unexported_symbols_list'))
        self.addOption(JoinedOrSeparateOption('-weak_framework'))
        self.headerpad_max_install_namesOption = self.addOption(JoinedOption('-headerpad_max_install_names'))
        self.twolevel_namespaceOption = self.addOption(FlagOption('-twolevel_namespace'))
        self.twolevel_namespace_hintsOption = self.addOption(FlagOption('-twolevel_namespace_hints'))
        self.prebindOption = self.addOption(FlagOption('-prebind'))
        self.noprebindOption = self.addOption(FlagOption('-noprebind'))
        self.nofixprebindingOption = self.addOption(FlagOption('-nofixprebinding'))
        self.prebind_all_twolevel_modulesOption = self.addOption(FlagOption('-prebind_all_twolevel_modules'))
        self.read_only_relocsOption = self.addOption(SeparateOption('-read_only_relocs'))
        self.addOption(FlagOption('-single_module'))
        self.nomultidefsOption = self.addOption(FlagOption('-nomultidefs'))
        self.nostartfilesOption = self.addOption(FlagOption('-nostartfiles'))
        self.nodefaultlibsOption = self.addOption(FlagOption('-nodefaultlibs'))
        self.nostdlibOption = self.addOption(FlagOption('-nostdlib'))
        self.addOption(FlagOption('-nostdinc'))
        self.objectOption = self.addOption(FlagOption('-object'))
        self.preloadOption = self.addOption(FlagOption('-preload'))
        self.staticOption = self.addOption(FlagOption('-static'))
        self.pagezero_sizeOption = self.addOption(FlagOption('-pagezero_size'))
        self.addOption(FlagOption('-shared'))
        self.staticLibgccOption = self.addOption(FlagOption('-static-libgcc'))
        self.sharedLibgccOption = self.addOption(FlagOption('-shared-libgcc'))
        self.addOption(FlagOption('-C'))
        self.addOption(FlagOption('-CC'))
        self.addOption(FlagOption('-R'))
        self.addOption(FlagOption('-P'))
        self.addOption(FlagOption('-all_load'))
        self.addOption(FlagOption('--constant-cfstrings'))
        self.addOption(FlagOption('-traditional'))
        self.addOption(FlagOption('--traditional'))
        self.addOption(FlagOption('-no_dead_strip_inits_and_terms'))
        self.whyloadOption = self.addOption(FlagOption('-whyload'))
        self.whatsloadedOption = self.addOption(FlagOption('-whatsloaded'))
        self.sectalignOption = self.addOption(MultiArgOption('-sectalign', numArgs=3))
        self.sectobjectsymbolsOption = self.addOption(MultiArgOption('-sectobjectsymbols', numArgs=2))
        self.segcreateOption = self.addOption(MultiArgOption('-segcreate', numArgs=3))
        self.segs_read_Option = self.addOption(JoinedOption('-segs_read_'))
        self.seglinkeditOption = self.addOption(FlagOption('-seglinkedit'))
        self.noseglinkeditOption = self.addOption(FlagOption('-noseglinkedit'))
        self.sectcreateOption = self.addOption(MultiArgOption('-sectcreate', numArgs=3))
        self.sectorderOption = self.addOption(MultiArgOption('-sectorder', numArgs=3))
        self.Zall_loadOption = self.addOption(FlagOption('-Zall_load'))
        self.Zallowable_clientOption = self.addOption(SeparateOption('-Zallowable_client'))
        self.Zbind_at_loadOption = self.addOption(SeparateOption('-Zbind_at_load'))
        self.ZbundleOption = self.addOption(FlagOption('-Zbundle'))
        self.Zbundle_loaderOption = self.addOption(JoinedOrSeparateOption('-Zbundle_loader'))
        self.Zdead_stripOption = self.addOption(FlagOption('-Zdead_strip'))
        self.Zdylib_fileOption = self.addOption(JoinedOrSeparateOption('-Zdylib_file'))
        self.ZdynamicOption = self.addOption(FlagOption('-Zdynamic'))
        self.ZdynamiclibOption = self.addOption(FlagOption('-Zdynamiclib'))
        self.Zexported_symbols_listOption = self.addOption(JoinedOrSeparateOption('-Zexported_symbols_list'))
        self.Zflat_namespaceOption = self.addOption(FlagOption('-Zflat_namespace'))
        self.Zfn_seg_addr_table_filenameOption = self.addOption(JoinedOrSeparateOption('-Zfn_seg_addr_table_filename'))
        self.Zforce_cpusubtype_ALLOption = self.addOption(FlagOption('-Zforce_cpusubtype_ALL'))
        self.Zforce_flat_namespaceOption = self.addOption(FlagOption('-Zforce_flat_namespace'))
        self.Zimage_baseOption = self.addOption(FlagOption('-Zimage_base'))
        self.ZinitOption = self.addOption(JoinedOrSeparateOption('-Zinit'))
        self.Zmulti_moduleOption = self.addOption(FlagOption('-Zmulti_module'))
        self.Zmultiply_definedOption = self.addOption(JoinedOrSeparateOption('-Zmultiply_defined'))
        self.ZmultiplydefinedunusedOption = self.addOption(JoinedOrSeparateOption('-Zmultiplydefinedunused'))
        self.ZmultiplydefinedunusedOption = self.addOption(JoinedOrSeparateOption('-Zmultiplydefinedunused'))
        self.Zno_dead_strip_inits_and_termsOption = self.addOption(FlagOption('-Zno_dead_strip_inits_and_terms'))
        self.Zseg_addr_tableOption = self.addOption(JoinedOrSeparateOption('-Zseg_addr_table'))
        self.ZsegaddrOption = self.addOption(JoinedOrSeparateOption('-Zsegaddr'))
        self.Zsegs_read_only_addrOption = self.addOption(JoinedOrSeparateOption('-Zsegs_read_only_addr'))
        self.Zsegs_read_write_addrOption = self.addOption(JoinedOrSeparateOption('-Zsegs_read_write_addr'))
        self.Zsingle_moduleOption = self.addOption(FlagOption('-Zsingle_module'))
        self.ZumbrellaOption = self.addOption(JoinedOrSeparateOption('-Zumbrella'))
        self.Zunexported_symbols_listOption = self.addOption(JoinedOrSeparateOption('-Zunexported_symbols_list'))
        self.Zweak_reference_mismatchesOption = self.addOption(JoinedOrSeparateOption('-Zweak_reference_mismatches'))

        # I dunno why these don't end up working when joined. Maybe
        # because of translation?
        self.filelistOption = self.addOption(SeparateOption('-filelist'))
        self.addOption(SeparateOption('-framework'))
        # FIXME: Alias.
        self.addOption(SeparateOption('-install_name'))
        self.Zinstall_nameOption = self.addOption(JoinedOrSeparateOption('-Zinstall_name'))
        self.addOption(SeparateOption('-seg_addr_table'))
        self.addOption(SeparateOption('-seg_addr_table_filename'))

        # Where are these coming from? I can't find them...
        self.eOption = self.addOption(JoinedOrSeparateOption('-e'))
        self.rOption = self.addOption(JoinedOrSeparateOption('-r'))

        # Is this actually declared anywhere? I can only find it in a
        # spec. :(
        self.pgOption = self.addOption(FlagOption('-pg'))

        doNotReallySupport = 1
        if doNotReallySupport:
            # Archaic gcc option.
            self.addOption(FlagOption('-cpp-precomp'))
            self.addOption(FlagOption('-no-cpp-precomp'))

        # C options for testing

        self.addOption(JoinedOrSeparateOption('-include'))
        self.AOption = self.addOption(SeparateOption('-A'))
        self.addOption(JoinedOrSeparateOption('-D'))
        self.FOption = self.addOption(JoinedOrSeparateOption('-F'))
        self.addOption(JoinedOrSeparateOption('-I'))
        self.LOption = self.addOption(JoinedOrSeparateOption('-L'))
        self.TOption = self.addOption(JoinedOrSeparateOption('-T'))
        self.addOption(JoinedOrSeparateOption('-U'))
        self.ZOption = self.addOption(JoinedOrSeparateOption('-Z'))

        self.addOption(JoinedOrSeparateOption('-l'))
        self.uOption = self.addOption(JoinedOrSeparateOption('-u'))
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
        self.dOption = self.addOption(FlagOption('-d'))
        self.addOption(JoinedOption('-d'))
        self.addOption(JoinedOption('-g'))

        self.f_exceptionsOption = self.addOption(FlagOption('-fexceptions'))
        self.f_objcOption = self.addOption(FlagOption('-fobjc'))
        self.f_openmpOption = self.addOption(FlagOption('-fopenmp'))
        self.f_gnuRuntimeOption = self.addOption(FlagOption('-fgnu-runtime'))
        self.f_nestedFunctionsOption = self.addOption(FlagOption('-fnested-functions'))
        self.f_pieOption = self.addOption(FlagOption('-fpie'))
        self.f_profileArcsOption = self.addOption(FlagOption('-fprofile-arcs'))
        self.f_profileGenerateOption = self.addOption(FlagOption('-fprofile-generate'))
        self.f_createProfileOption = self.addOption(FlagOption('-fcreate-profile'))
        self.coverageOption = self.addOption(FlagOption('-coverage'))
        self.coverageOption2 = self.addOption(FlagOption('--coverage'))
        self.addOption(JoinedOption('-f'))

        self.m_32Option = self.addOption(FlagOption('-m32'))
        self.m_64Option = self.addOption(FlagOption('-m64'))
        self.m_iphoneosVersionMinOption = self.addOption(JoinedOption('-miphoneos-version-min='))
        self.m_macosxVersionMinOption = self.addOption(JoinedOption('-mmacosx-version-min='))

        # Ugh. Need to disambiguate our naming convetion. -m x goes to
        # the linker sometimes, wheres -mxxxx is used for a variety of
        # other things.
        self.mOption = self.addOption(SeparateOption('-m'))
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
