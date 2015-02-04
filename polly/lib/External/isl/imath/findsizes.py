#!/usr/bin/env python
##
## Name:     findsizes.py
## Purpose:  Find acceptable digit and word types for IMath.
## Author:   M. J. Fromberger <http://spinning-yarns.org/michael/>
##
##  Copyright (C) 2002-2008 Michael J. Fromberger, All Rights Reserved.
##
##  Permission is hereby granted, free of charge, to any person obtaining a
##  copy of this software and associated documentation files (the "Software"),
##  to deal in the Software without restriction, including without limitation
##  the rights to use, copy, modify, merge, publish, distribute, sublicense,
##  and/or sell copies of the Software, and to permit persons to whom the
##  Software is furnished to do so, subject to the following conditions:
##
##  The above copyright notice and this permission notice shall be included in
##  all copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
##  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
##  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
##  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
##  DEALINGS IN THE SOFTWARE.
##
import getopt, os, re, subprocess, sys, tempfile

# These are the type names to test for suitability.  If your compiler
# does not support "long long", e.g., it is strict ANSI C90, then you
# should remove "long long" from this list.
try_types = [ "unsigned %s" % s for s in
              ("char", "short", "int", "long", "long long") ]

def main(args):
    """Scan the Makefile to find appropriate compiler settings, and then
    compile a test program to emit the sizes of the various types that are
    considered candidates.  The -L (--nolonglong) command line option disables
    the use of the"long long" type, which is not standard ANSI C90; by default,
    "long long" is considered.
    """
    try:
        opts, args = getopt.getopt(args, 'L', ('nolonglong',))
    except getopt.GetoptError, e:
        print >> sys.stderr, "Usage: findsizes.py [-L/--nolonglong]"
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-L', '--nolonglong'):
            try:
                try_types.pop(try_types.index("unsigned long long"))
            except ValueError: pass
    
    vars = get_make_info()
    sizes = get_type_sizes(try_types, vars.get('CC', 'cc'),
                           vars.get('CFLAGS', '').split())
    
    stypes = sorted(sizes.keys(), key = lambda k: sizes[k], reverse = True)
    word_type = stypes[0]
    for t in stypes[1:]:
        if sizes[t] <= sizes[word_type] / 2:
            digit_type = t
            break
    else:
        print >> sys.stderr, "Unable to find a compatible digit type."
        sys.exit(1)

    print "typedef %-20s mp_word;   /* %d bytes */\n" \
          "typedef %-20s mp_digit;  /* %d bytes */" % \
          (word_type, sizes[word_type], digit_type, sizes[digit_type])

def get_type_sizes(types, cc, cflags = ()):
    """Return a dictionary mapping the names of the specified types to
    their sizes in bytes, based on the output of the C compiler whose
    path and arguments are given.
    """
    fd, tpath = tempfile.mkstemp(suffix = '.c')
    fp = os.fdopen(fd, 'r+')
    fp.seek(0)
    
    fp.write("#include <stdio.h>\n\nint main(void)\n{\n")
    for t in types:
        fp.write('  printf("%%lu\\t%%s\\n", (unsigned long) sizeof(%s), '
                 '\"%s\");\n' % (t, t))
    fp.write('\n  return 0;\n}\n')
    fp.close()

    print >> sys.stderr, \
          "Compiler:  %s\n" \
          "Flags:     %s\n" \
          "Source:    %s\n" % (cc, ' '.join(cflags), tpath)
    
    cmd = [cc] + list(cflags) + [tpath]
    if subprocess.call(cmd) <> 0:
        raise ValueError("Error while running '%s'" % ' '.join(cmd))

    os.unlink(tpath)
    if not os.path.isfile('a.out'):
        raise ValueError("No executable a.out found")

    result = subprocess.Popen(['./a.out'],
                              stdout = subprocess.PIPE).communicate()[0]

    out = {}
    for line in result.split('\n'):
        if line.strip() == '':
            continue
        size, type = re.split(r'\s+', line, 1)
        out[type] = int(size)
    
    os.unlink("a.out")
    return out

def sub_make_vars(input, vars):
    """Perform Make style variable substitution in the given input
    string, using vars as a dictionary of variables to substitute.
    """
    def frep(m):
        try:
            return vars[m.group(1).strip()]
        except KeyError:
            return ' '
    
    expr = re.compile(r'\$\((\s*\w+\s*)\)')
    out = input
    while True:
        next = expr.sub(frep, out)
        if next == out:
            break
        out = next
    
    return out

def get_make_info(target = None, makefile = None, makepath = "make",
                  defs = ()):
    """Extract a listing of all of the variables defined by Make.
    Returns a dictionary mapping variable names to their string
    values.

    Optional arguments:
    target   -- the name of the target to request that Make build.
    makefile -- the path to the Makefile.
    makepath -- the path to the make executable.
    defs     -- a sequence of strings, additional make arguments.
    """
    cmd = [makepath]
    if defs:
        cmd.extend(defs)
    cmd.extend(("-p", "-n"))
    if makefile is not None:
        cmd.extend(["-f", makefile])
    if target is not None:
        cmd.append(target)
    
    output = subprocess.Popen(cmd,
                              stdout = subprocess.PIPE,
                              stderr = subprocess.PIPE).communicate()[0]
    vrule = re.compile(r'^(\w+)\s*=\s*(.*)$')
    
    vars = {}
    for line in output.split('\n'):
        m = vrule.match(line)
        if m:
            vars[m.group(1)] = m.group(2)

    for key, val in vars.iteritems():
        vars[key] = sub_make_vars(val, vars)
    
    return vars

if __name__ == "__main__":
    main(sys.argv[1:])

# Here there be dragons
