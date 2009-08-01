import errno, os, sys

def warning(msg):
    print >>sys.stderr, '%s: warning: %s' % (sys.argv[0], msg)

def detectCPUs():
    """
    Detects the number of CPUs on a system. Cribbed from pp.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else: # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
        if ncpus > 0:
            return ncpus
    return 1 # Default

def mkdir_p(path):
    """mkdir_p(path) - Make the "path" directory, if it does not exist; this
    will also make directories for any missing parent directories."""

    if not path or os.path.exists(path):
        return

    parent = os.path.dirname(path) 
    if parent != path:
        mkdir_p(parent)

    try:
        os.mkdir(path)
    except OSError,e:
        # Ignore EEXIST, which may occur during a race condition.
        if e.errno != errno.EEXIST:
            raise

def which(command, paths = None):
    """which(command, [paths]) - Look up the given command in the paths string (or
    the PATH environment variable, if unspecified)."""

    if paths is None:
        paths = os.environ.get('PATH','')

    # Check for absolute match first.
    if os.path.exists(command):
        return command

    # Would be nice if Python had a lib function for this.
    if not paths:
        paths = os.defpath

    # Get suffixes to search.
    pathext = os.environ.get('PATHEXT', '').split(os.pathsep)

    # Search the paths...
    for path in paths.split(os.pathsep):
        for ext in pathext:
            p = os.path.join(path, command + ext)
            if os.path.exists(p):
                return p

    return None

