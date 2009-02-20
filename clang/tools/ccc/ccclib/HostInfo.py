import ToolChain

class HostInfo(object):
    """HostInfo - Config information about a particular host which may
    interact with driver behavior. This can be very different from the
    target(s) of a particular driver invocation."""

    def __init__(self, driver):
        self.driver = driver

    def getArchName(self, args):
        abstract

    def useDriverDriver(self):
        abstract

    def getToolChain(self):
        abstract

    def getToolChainForArch(self, arch):
        raise RuntimeError,"getToolChainForArch() unsupported on this host."

# Darwin

class DarwinHostInfo(HostInfo):
    def __init__(self, driver):
        super(DarwinHostInfo, self).__init__(driver)
        
        # FIXME: Find right regex for this.
        import re
        m = re.match(r'([0-9]+)\.([0-9]+)\.([0-9]+)', driver.getHostReleaseName())
        if not m:
            raise RuntimeError,"Unable to determine Darwin version."
        self.darwinVersion = tuple(map(int, m.groups()))
        self.gccVersion = (4,2,1)

    def useDriverDriver(self):
        return True

    def getToolChain(self):
        return self.getToolChainForArch(self.getArchName(None))

    def getToolChainForArch(self, arch):
        if arch in ('i386', 'x86_64'):
            return ToolChain.Darwin_X86_ToolChain(self.driver,
                                                  arch,
                                                  self.darwinVersion,
                                                  self.gccVersion)

        return ToolChain.Generic_GCC_ToolChain(self.driver, '')

class DarwinPPCHostInfo(DarwinHostInfo):
    def getArchName(self, args):
        if args and args.getLastArg(args.parser.m_64Option):
            return 'ppc64'
        return 'ppc'

class DarwinPPC_64HostInfo(DarwinHostInfo):
    def getArchName(self, args):
        if args and args.getLastArg(args.parser.m_32Option):
            return 'ppc'
        return 'ppc64'

class DarwinX86HostInfo(DarwinHostInfo):
    def getArchName(self, args):
        if args and args.getLastArg(args.parser.m_64Option):
            return 'x86_64'
        return 'i386'

class DarwinX86_64HostInfo(DarwinHostInfo):
    def getArchName(self, args):
        if args and args.getLastArg(args.parser.m_32Option):
            return 'i386'
        return 'x86_64'

def getDarwinHostInfo(driver):
    machine = driver.getHostMachine()
    bits = driver.getHostBits()
    if machine == 'i386':
        if bits == '32':
            return DarwinX86HostInfo(driver)
        if bits == '64':
            return DarwinX86_64HostInfo(driver)
    elif machine == 'ppc':
        if bits == '32':
            return DarwinPPCHostInfo(driver)
        if bits == '64':
            return DarwinPPC_64HostInfo(driver)
            
    raise RuntimeError,'Unrecognized Darwin platform: %r:%r' % (machine, bits)

# Unknown

class UnknownHostInfo(HostInfo):
    def getArchName(self, args):
        raise RuntimeError,'getArchName() unsupported on unknown host.'

    def useDriverDriver(self):
        return False

    def getToolChain(self):
        return ToolChain.Generic_GCC_ToolChain(self.driver, '')

def getUnknownHostInfo(driver):
    return UnknownHostInfo(driver)

####

kSystems = {
    'darwin' : getDarwinHostInfo,
    'unknown' : getUnknownHostInfo,
    }

def getHostInfo(driver):
    system = driver.getHostSystemName()
    handler = kSystems.get(system)
    if handler:
        return handler(driver)

    driver.warning('Unknown host %r, using generic host information.' % system)
    return UnknownHostInfo(driver)
