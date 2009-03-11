import os

import Arguments
import Phases
import Tools
import Types

###

class ToolChain(object):
    """ToolChain - Provide mappings of Actions to Tools."""

    def __init__(self, driver, archName,
                 filePathPrefixes=[],
                 programPathPrefixes=[]):
        self.driver = driver
        self.archName = archName
        self.filePathPrefixes = list(filePathPrefixes)
        self.programPathPrefixes = list(programPathPrefixes)

    def getFilePath(self, name):
        return self.driver.getFilePath(name, self)
        
    def getProgramPath(self, name):
        return self.driver.getProgramPath(name, self)

    def selectTool(self, action):
        """selectTool - Return a Tool instance to use for handling
        some particular action."""
        abstract

    def translateArgs(self, args, arch):
        """translateArgs - Callback to allow argument translation for
        an entire toolchain."""

        # FIXME: Would be nice to move arch handling out of generic
        # code.
        if arch:
            archName = args.getValue(arch)
            al = Arguments.DerivedArgList(args)
            for arg in args.args:
                if arg.opt is args.parser.archOption:
                    if arg is arch:
                        al.append(arg)
                elif arg.opt is args.parser.XarchOption:
                    if args.getJoinedValue(arg) == archName:
                        # FIXME: Fix this.
                        arg = args.parser.lookupOptForArg(Arguments.InputIndex(0, arg.index.pos + 1),
                                                          args.getSeparateValue(arg),
                                                          iter([]))
                        al.append(arg)
                else:
                    al.append(arg)
            return al
        else:
            return args

    def shouldUseClangCompiler(self, action):
        # If user requested no clang, or this isn't a "compile" phase,
        # or this isn't an input clang understands, then don't use clang.
        if (self.driver.cccNoClang or 
            not isinstance(action.phase, (Phases.PreprocessPhase,
                                          Phases.CompilePhase,
                                          Phases.SyntaxOnlyPhase,
                                          Phases.EmitLLVMPhase,
                                          Phases.PrecompilePhase)) or
            action.inputs[0].type not in Types.clangableTypesSet):
            return False

        if self.driver.cccNoClangPreprocessor:
            if isinstance(action.phase, Phases.PreprocessPhase):
                return False

        if self.driver.cccNoClangCXX:
            if action.inputs[0].type in Types.cxxTypesSet:
                return False
        
        # Don't use clang if this isn't one of the user specified
        # archs to build.
        if (self.driver.cccClangArchs and 
            self.archName not in self.driver.cccClangArchs):
            return False

        return True
        
    def isMathErrnoDefault(self):
        return True

    def isUnwindTablesDefault(self):
        # FIXME: Target hook.
        if self.archName == 'x86_64':
            return True
        return False

    def getDefaultRelocationModel(self):
        return 'static'
    
    def getForcedPicModel(self):
        return 

class Darwin_X86_ToolChain(ToolChain):
    def __init__(self, driver, archName, darwinVersion, gccVersion):
        super(Darwin_X86_ToolChain, self).__init__(driver, archName)
        assert isinstance(darwinVersion, tuple) and len(darwinVersion) == 3
        assert isinstance(gccVersion, tuple) and len(gccVersion) == 3
        self.darwinVersion = darwinVersion
        self.gccVersion = gccVersion

        self.clangTool = Tools.Clang_CompileTool(self)
        cc = Tools.Darwin_X86_CompileTool(self)
        self.toolMap = {
            Phases.PreprocessPhase : Tools.Darwin_X86_PreprocessTool(self),
            Phases.AnalyzePhase : self.clangTool,
            Phases.SyntaxOnlyPhase : cc,
            Phases.EmitLLVMPhase : cc,
            Phases.CompilePhase : cc,
            Phases.PrecompilePhase : cc,
            Phases.AssemblePhase : Tools.Darwin_AssembleTool(self),
            Phases.LinkPhase : Tools.Darwin_X86_LinkTool(self),
            Phases.LipoPhase : Tools.LipoTool(self),
            }

        if archName == 'x86_64':
            self.filePathPrefixes.append(os.path.join(self.driver.driverDir,
                                                      '../lib/gcc',
                                                      self.getToolChainDir(),
                                                      'x86_64'))
            self.filePathPrefixes.append(os.path.join('/usr/lib/gcc',
                                                      self.getToolChainDir(),
                                                      'x86_64'))
        self.filePathPrefixes.append(os.path.join(self.driver.driverDir,
                                                  '../lib/gcc',
                                                  self.getToolChainDir()))
        self.filePathPrefixes.append(os.path.join('/usr/lib/gcc',
                                                  self.getToolChainDir()))

        self.programPathPrefixes.append(os.path.join(self.driver.driverDir,
                                                  '../libexec/gcc',
                                                  self.getToolChainDir()))
        self.programPathPrefixes.append(os.path.join('/usr/libexec/gcc',
                                                     self.getToolChainDir()))
        self.programPathPrefixes.append(self.driver.driverDir)

    def getToolChainDir(self):
        return 'i686-apple-darwin%d/%s' % (self.darwinVersion[0],
                                           '.'.join(map(str,self.gccVersion)))

    def getMacosxVersionMin(self):
        major,minor,minorminor = self.darwinVersion
        return '%d.%d.%d' % (10, major-4, minor)

    def selectTool(self, action):
        assert isinstance(action, Phases.JobAction)
        
        if self.shouldUseClangCompiler(action):
            return self.clangTool

        return self.toolMap[action.phase.__class__]

    def translateArgs(self, args, arch):
        args = super(Darwin_X86_ToolChain, self).translateArgs(args, arch)
        
        # If arch hasn't been bound we don't need to do anything yet.
        if not arch:
            return args

        # FIXME: We really want to get out of the tool chain level
        # argument translation business, as it makes the driver
        # functionality much more opaque. For now, we follow gcc
        # closely solely for the purpose of easily achieving feature
        # parity & testability. Once we have something that works, we
        # should reevaluate each translation and try to push it down
        # into tool specific logic.

        al = Arguments.DerivedArgList(args)
        if not args.getLastArg(args.parser.m_macosxVersionMinOption):
            al.append(al.makeJoinedArg(self.getMacosxVersionMin(),
                                       args.parser.m_macosxVersionMinOption))
        for arg in args:
            # Sob. These is strictly gcc compatible for the time
            # being. Apple gcc translates options twice, which means
            # that self-expanding options add duplicates.
            if arg.opt is args.parser.m_kernelOption:
                al.append(arg)
                al.append(al.makeFlagArg(args.parser.staticOption))
                al.append(al.makeFlagArg(args.parser.staticOption))
            elif arg.opt is args.parser.dependencyFileOption:
                al.append(al.makeSeparateArg(args.getValue(arg),
                                             args.parser.MFOption))
            elif arg.opt is args.parser.gfullOption:
                al.append(al.makeFlagArg(args.parser.gOption))
                al.append(al.makeFlagArg(args.parser.f_noEliminateUnusedDebugSymbolsOption))
            elif arg.opt is args.parser.gusedOption:
                al.append(al.makeFlagArg(args.parser.gOption))
                al.append(al.makeFlagArg(args.parser.f_eliminateUnusedDebugSymbolsOption))
            elif arg.opt is args.parser.f_appleKextOption:
                al.append(arg)
                al.append(al.makeFlagArg(args.parser.staticOption))
                al.append(al.makeFlagArg(args.parser.staticOption))
            elif arg.opt is args.parser.f_terminatedVtablesOption:
                al.append(al.makeFlagArg(args.parser.f_appleKextOption))
                al.append(al.makeFlagArg(args.parser.staticOption))
            elif arg.opt is args.parser.f_indirectVirtualCallsOption:
                al.append(al.makeFlagArg(args.parser.f_appleKextOption))
                al.append(al.makeFlagArg(args.parser.staticOption))
            elif arg.opt is args.parser.sharedOption:
                al.append(al.makeFlagArg(args.parser.dynamiclibOption))
            elif arg.opt is args.parser.f_constantCfstringsOption:
                al.append(al.makeFlagArg(args.parser.m_constantCfstringsOption))
            elif arg.opt is args.parser.f_noConstantCfstringsOption:
                al.append(al.makeFlagArg(args.parser.m_noConstantCfstringsOption))
            elif arg.opt is args.parser.WnonportableCfstringsOption:
                al.append(al.makeFlagArg(args.parser.m_warnNonportableCfstringsOption))
            elif arg.opt is args.parser.WnoNonportableCfstringsOption:
                al.append(al.makeFlagArg(args.parser.m_noWarnNonportableCfstringsOption))
            elif arg.opt is args.parser.f_pascalStringsOption:
                al.append(al.makeFlagArg(args.parser.m_pascalStringsOption))
            elif arg.opt is args.parser.f_noPascalStringsOption:
                al.append(al.makeFlagArg(args.parser.m_noPascalStringsOption))
            else:
                al.append(arg)

        # FIXME: Actually, gcc always adds this, but it is filtered
        # for duplicates somewhere. This also changes the order of
        # things, so look it up.
        if arch and args.getValue(arch) == 'x86_64':
            if not args.getLastArg(args.parser.m_64Option):
                al.append(al.makeFlagArg(args.parser.m_64Option))

        if not args.getLastArg(args.parser.m_tuneOption):
            al.append(al.makeJoinedArg('core2',
                                       args.parser.m_tuneOption))

        return al

    def isMathErrnoDefault(self):
        return False

    def getDefaultRelocationModel(self):
        return 'pic'
    
    def getForcedPicModel(self):
        if self.archName == 'x86_64':
            return 'pic'

class Generic_GCC_ToolChain(ToolChain):
    """Generic_GCC_ToolChain - A tool chain using the 'gcc' command to
    perform all subcommands; this relies on gcc translating the
    options appropriately."""

    def __init__(self, driver, archName):
        super(Generic_GCC_ToolChain, self).__init__(driver, archName)
        cc = Tools.GCC_CompileTool(self)
        self.clangTool = Tools.Clang_CompileTool(self)
        self.toolMap = {
            Phases.PreprocessPhase : Tools.GCC_PreprocessTool(self),
            Phases.AnalyzePhase : self.clangTool,
            Phases.SyntaxOnlyPhase : cc,
            Phases.EmitLLVMPhase : cc,
            Phases.CompilePhase : cc,
            Phases.PrecompilePhase : Tools.GCC_PrecompileTool(self),
            Phases.AssemblePhase : Tools.GCC_AssembleTool(self),
            Phases.LinkPhase : Tools.GCC_LinkTool(self),
            }

    def selectTool(self, action):
        assert isinstance(action, Phases.JobAction)

        if self.shouldUseClangCompiler(action):
            return self.clangTool

        return self.toolMap[action.phase.__class__]

class Darwin_GCC_ToolChain(Generic_GCC_ToolChain):
    def getRelocationModel(self, picEnabled, picDisabled):
        if picEnabled:
            return 'pic'
        elif picDisabled:
            return 'static'
        else:
            return 'dynamic-no-pic'

