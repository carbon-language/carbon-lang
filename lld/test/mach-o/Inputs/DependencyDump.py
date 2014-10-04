# -*- Python -*-


#
# Dump out Xcode binary dependency file.
#

import sys

f = open(sys.argv[1], "rb")
byte = f.read(1)
while byte != '':
    if byte == '\000':
        sys.stdout.write("linker-vers: ")
    elif byte == '\020':
        sys.stdout.write("input-file:  ")
    elif byte == '\021':
        sys.stdout.write("not-found:   ")
    elif byte == '\100':
        sys.stdout.write("output-file: ")
    byte = f.read(1)
    while byte != '\000':
        if byte != '\012':
            sys.stdout.write(byte)
        byte = f.read(1)
    sys.stdout.write("\n")
    byte = f.read(1)

f.close()

