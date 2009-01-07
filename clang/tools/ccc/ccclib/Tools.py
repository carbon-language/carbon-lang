import Arguments
import Jobs
import Types

class Tool(object):
    """Tool - A concrete implementation of an action."""

    eFlagsPipedInput = 1 << 0
    eFlagsPipedOutput = 1 << 1
    eFlagsIntegratedCPP = 1 << 2

    def __init__(self, name, flags = 0):
        self.name = name
        self.flags = flags

    def acceptsPipedInput(self):
        return not not (self.flags & Tool.eFlagsPipedInput)
    def canPipeOutput(self):
        return not not (self.flags & Tool.eFlagsPipedOutput)
    def hasIntegratedCPP(self):
        return not not (self.flags & Tool.eFlagsIntegratedCPP)

class GCC_Common_Tool(Tool):
    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args,
                     extraArgs):
        assert len(inputs) == 1

        input = inputs[0]

        cmd_args = args + extraArgs
        if arch:
            # FIXME: Clean this up.
            if isinstance(arch, Arguments.DerivedArg):
                cmd_args.extend([Arguments.DerivedArg('-arch'),
                                 arch])
            else:
                cmd_args.append(arch)
        if isinstance(output, Jobs.PipedJob):
            cmd_args.extend([Arguments.DerivedArg('-o'), Arguments.DerivedArg('-')])
        elif output is None:
            cmd_args.append(Arguments.DerivedArg('-fsyntax-only'))
        else:
            # FIXME: Ditch this hack.
            if isinstance(output, Arguments.DerivedArg):
                cmd_args.extend([Arguments.DerivedArg('-o'), output])
            else:
                cmd_args.append(output)

        cmd_args.extend([Arguments.DerivedArg('-x'),
                         Arguments.DerivedArg(input.type.name)])
        if isinstance(input.source, Jobs.PipedJob):
            cmd_args.append(Arguments.DerivedArg('-'))
        else:
            cmd_args.append(input.source)

        jobs.addJob(Jobs.Command('gcc', cmd_args))

class GCC_PreprocessTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_PreprocessTool, self).__init__('gcc',
                                                 (Tool.eFlagsPipedInput |
                                                  Tool.eFlagsPipedOutput))

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args):
        return super(GCC_PreprocessTool, self).constructJob(phase, arch, jobs, inputs,
                                                            output, outputType, args,
                                                            [Arguments.DerivedArg('-E')])

class GCC_CompileTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_CompileTool, self).__init__('gcc',
                                              (Tool.eFlagsPipedInput |
                                               Tool.eFlagsPipedOutput |
                                               Tool.eFlagsIntegratedCPP))

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args):
        return super(GCC_CompileTool, self).constructJob(phase, arch, jobs, inputs,
                                                         output, outputType, args,
                                                         [Arguments.DerivedArg('-S')])

class GCC_PrecompileTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_PrecompileTool, self).__init__('gcc',
                                                 (Tool.eFlagsPipedInput |
                                                  Tool.eFlagsIntegratedCPP))

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args):
        return super(GCC_PrecompileTool, self).constructJob(phase, arch, jobs, inputs,
                                                            output, outputType, args,
                                                            [])

class DarwinAssemblerTool(Tool):
    def __init__(self):
        super(DarwinAssemblerTool, self).__init__('as',
                                                  Tool.eFlagsPipedInput)

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args):
        assert len(inputs) == 1
        assert outputType is Types.ObjectType

        input = inputs[0]

        cmd_args = []
        if arch:
            # FIXME: Clean this up.
            if isinstance(arch, Arguments.DerivedArg):
                cmd_args.extend([Arguments.DerivedArg('-arch'),
                                 arch])
            else:
                cmd_args.append(arch)
        cmd_args.append(Arguments.DerivedArg('-force_cpusubtype_ALL'))
        if isinstance(output, Arguments.DerivedArg):
            cmd_args.extend([Arguments.DerivedArg('-o'), output])
        else:
            cmd_args.append(output)
        if isinstance(input.source, Jobs.PipedJob):
            cmd_args.append(Arguments.DerivedArg('-'))
        else:
            cmd_args.append(input.source)
        jobs.addJob(Jobs.Command('as', cmd_args))

class Collect2Tool(Tool):
    kCollect2Path = '/usr/libexec/gcc/i686-apple-darwin10/4.2.1/collect2'
    def __init__(self):
        super(Collect2Tool, self).__init__('collect2')

    def constructJob(self, phase, arch, jobs, inputs,
                     output, outputType, args):
        assert outputType is Types.ImageType

        cmd_args = []
        for arg in args:
            if arg.opt:
                if arg.opt.name in ('-framework',):
                    cmd_args.append(arg)
        for input in inputs:
            cmd_args.append(input.source)
        if isinstance(output, Arguments.DerivedArg):
            cmd_args.extend([Arguments.DerivedArg('-o'), output])
        else:
            cmd_args.append(output)
        cmd_args.extend([Arguments.DerivedArg('-L/usr/lib/gcc/i686-apple-darwin10/4.2.1'),
                         Arguments.DerivedArg('-lcrt1.10.5.o'),
                         Arguments.DerivedArg('-lgcc_s.10.5'),
                         Arguments.DerivedArg('-lgcc'),
                         Arguments.DerivedArg('-lSystem')])
        jobs.addJob(Jobs.Command(self.kCollect2Path, cmd_args))

class LipoTool(Tool):
    def __init__(self):
        super(LipoTool, self).__init__('lipo')

    def constructJob(self, phase, arch, jobs, inputs,
                     output, outputType, args):
        assert outputType is Types.ImageType

        cmd_args = [Arguments.DerivedArg('-create')]
        if isinstance(output, Arguments.DerivedArg):
            cmd_args.extend([Arguments.DerivedArg('-o'), output])
        else:
            cmd_args.append(output)
        for input in inputs:
            cmd_args.append(input.source)
        jobs.addJob(Jobs.Command('lipo', cmd_args))
