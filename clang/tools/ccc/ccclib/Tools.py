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
                     output, outputType, args, arglist,
                     extraArgs):
        cmd_args = sum(map(arglist.render, args),[]) + extraArgs
        if arch:
            cmd_args.extend(arglist.render(arch))
        if isinstance(output, Jobs.PipedJob):
            cmd_args.extend(['-o', '-'])
        elif output is None:
            cmd_args.append('-fsyntax-only')
        else:
            cmd_args.extend(arglist.render(output))

        # Only pass -x if gcc will understand it; otherwise hope gcc
        # understands the suffix correctly. The main use case this
        # would go wrong in is for linker inputs, say if the user
        # tried to make an executable named 'a.c'.
        #
        # FIXME: For the linker case specifically, can we safely
        # convert inputs into '-Wl,' options?
        for input in inputs:
            if input.type.canBeUserSpecified:
                cmd_args.extend(['-x', input.type.name])

            if isinstance(input.source, Jobs.PipedJob):
                cmd_args.append('-')
            else:
                cmd_args.append(arglist.getValue(input.source))

        jobs.addJob(Jobs.Command('gcc', cmd_args))

class GCC_PreprocessTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_PreprocessTool, self).__init__('gcc (cpp)',
                                                 (Tool.eFlagsPipedInput |
                                                  Tool.eFlagsPipedOutput))

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_PreprocessTool, self).constructJob(phase, arch, jobs, inputs,
                                                            output, outputType, args, arglist,
                                                            ['-E'])

class GCC_CompileTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_CompileTool, self).__init__('gcc (cc1)',
                                              (Tool.eFlagsPipedInput |
                                               Tool.eFlagsPipedOutput |
                                               Tool.eFlagsIntegratedCPP))

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_CompileTool, self).constructJob(phase, arch, jobs, inputs,
                                                         output, outputType, args, arglist,
                                                         ['-S'])

class GCC_PrecompileTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_PrecompileTool, self).__init__('gcc (pch)',
                                                 (Tool.eFlagsPipedInput |
                                                  Tool.eFlagsIntegratedCPP))

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_PrecompileTool, self).constructJob(phase, arch, jobs, inputs,
                                                            output, outputType, args, arglist,
                                                            [])

class DarwinAssembleTool(Tool):
    def __init__(self):
        super(DarwinAssembleTool, self).__init__('as',
                                                 Tool.eFlagsPipedInput)

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        assert len(inputs) == 1
        assert outputType is Types.ObjectType

        input = inputs[0]

        cmd_args = []
        if arch:
            cmd_args.extend(arglist.render(arch))
        cmd_args.append('-force_cpusubtype_ALL')
        cmd_args.extend(arglist.render(output))
        if isinstance(input.source, Jobs.PipedJob):
            cmd_args.append('-')
        else:
            cmd_args.append(arglist.getValue(input.source))
        jobs.addJob(Jobs.Command('as', cmd_args))

class GCC_AssembleTool(GCC_Common_Tool):
    def __init__(self):
        # We can't generally assume the assembler can take or output
        # on pipes.
        super(GCC_AssembleTool, self).__init__('gcc (as)')

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_AssembleTool, self).constructJob(phase, arch, jobs, inputs,
                                                          output, outputType, args, arglist,
                                                          ['-c'])

class GCC_LinkTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_LinkTool, self).__init__('gcc (ld)')

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_LinkTool, self).constructJob(phase, arch, jobs, inputs,
                                                      output, outputType, args, arglist,
                                                      [])

class Collect2Tool(Tool):
    kCollect2Path = '/usr/libexec/gcc/i686-apple-darwin10/4.2.1/collect2'
    def __init__(self):
        super(Collect2Tool, self).__init__('collect2')

    def constructJob(self, phase, arch, jobs, inputs,
                     output, outputType, args, arglist):
        assert outputType is Types.ImageType

        cmd_args = []
        for arg in args:
            if arg.opt.name in ('-framework',):
                cmd_args.extend(arglist.render(arg))
        for input in inputs:
            cmd_args.append(arglist.getValue(input.source))
        cmd_args.extend(arglist.render(output))
        cmd_args.extend(['-L/usr/lib/gcc/i686-apple-darwin10/4.2.1',
                         '-lcrt1.10.5.o',
                         '-lgcc_s.10.5',
                         '-lgcc',
                         '-lSystem'])
        jobs.addJob(Jobs.Command(self.kCollect2Path, cmd_args))

class LipoTool(Tool):
    def __init__(self):
        super(LipoTool, self).__init__('lipo')

    def constructJob(self, phase, arch, jobs, inputs,
                     output, outputType, args, arglist):
        assert outputType is Types.ImageType

        cmd_args = ['-create']
        cmd_args.extend(arglist.render(output))
        for input in inputs:
            cmd_args.append(arglist.getValue(input.source))
        jobs.addJob(Jobs.Command('lipo', cmd_args))
