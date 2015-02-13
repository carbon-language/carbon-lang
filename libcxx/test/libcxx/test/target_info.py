import locale
import platform
import sys

class TargetInfo(object):
    def platform(self):
        raise NotImplementedError

    def system(self):
        raise NotImplementedError

    def platform_ver(self):
        raise NotImplementedError

    def platform_name(self):
        raise NotImplementedError

    def supports_locale(self, loc):
        raise NotImplementedError


class LocalTI(TargetInfo):
    def platform(self):
        platform_name = sys.platform.lower().strip()
        # Strip the '2' from linux2.
        if platform_name.startswith('linux'):
            platform_name = 'linux'
        return platform_name

    def system(self):
        return platform.system()

    def platform_name(self):
        if platform() == 'linux':
            name, _, _ = platform.linux_distribution()
            name = name.lower().strip()
            if name:
                return name
        return None

    def platform_ver(self):
        if platform() == 'linux':
            _, ver, _ = platform.linux_distribution()
            ver = ver.lower().strip()
            if ver:
                return ver
        return None

    def supports_locale(self, loc):
        try:
            locale.setlocale(locale.LC_ALL, loc)
            return True
        except locale.Error:
            return False

