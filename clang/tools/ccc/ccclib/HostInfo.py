class HostInfo(object):
    """HostInfo - Config information about a particular host which may
    interact with driver behavior. This can be very different from the
    target(s) of a particular driver invocation."""

    def __init__(self):
        pass

    def getArchName(self):
        abstract

    def useDriverDriver(self):
        abstract

# Darwin

class DarwinHostInfo(HostInfo):
    def useDriverDriver(self):
        return True

class DarwinPPCHostInfo(DarwinHostInfo):
    def getArchName(self):
        return 'ppc'

class DarwinPPC_64HostInfo(DarwinHostInfo):
    def getArchName(self):
        return 'ppc64'

class DarwinX86HostInfo(DarwinHostInfo):
    def getArchName(self):
        return 'i386'

class DarwinX86_64HostInfo(DarwinHostInfo):
    def getArchName(self):
        return 'x86_64'

def getDarwinHostInfo(driver):
    machine = driver.getHostMachine()
    bits = driver.getHostBits()
    if machine == 'i386':
        if bits == '32':
            return DarwinX86HostInfo()
        if bits == '64':
            return DarwinX86_64HostInfo()
    elif machine == 'ppc':
        if bits == '32':
            return DarwinPPCHostInfo()
        if bits == '64':
            return DarwinPPC_64HostInfo()
            
    raise RuntimeError,'Unrecognized Darwin platform: %r:%r' % (machine, bits)

# Unknown

class UnknownHostInfo(HostInfo):
    def getArchName(self):
        raise RuntimeError,'getArchName() unsupported on unknown host.'

    def useDriverDriver(self):
        return False

def getUnknownHostInfo(driver):
    return UnknownHostInfo()

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
    return UnknownHostInfo()
