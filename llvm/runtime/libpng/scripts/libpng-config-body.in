
usage()
{
    cat <<EOF
Usage: libpng-config [OPTION] ...

Known values for OPTION are:

  --prefix        print libpng prefix
  --libdir        print path to directory containing library
  --libs          print library linking information
  --ccopts        print compiler options
  --cppflags      print pre-processor flags
  --cflags        print preprocessor flags, I_opts, and compiler options
  --I_opts        print "-I" include options
  --L_opts        print linker "-L" flags for dynamic linking
  --R_opts        print dynamic linker "-R" or "-rpath" flags
  --ldopts        print linker options
  --ldflags       print linker flags (ldopts, L_opts, R_opts, and libs)
  --static        revise subsequent outputs for static linking
  --help          print this help and exit
  --version       print version information
EOF

    exit $1
}

if test $# -eq 0; then
    usage 1
fi

while test $# -gt 0; do
    case "$1" in

    --prefix)
        echo ${prefix}
        ;;

    --version)
        echo ${version}
        exit 0
        ;;

    --help)
        usage 0
        ;;

    --ccopts)
        echo ${ccopts}
        ;;

    --cppflags)
        echo ${cppflags}
        ;;

    --cflags)
        echo ${I_opts} ${cppflags} ${ccopts}
        ;;

    --libdir)
        echo ${libdir}
        ;;

    --libs)
        echo ${libs}
        ;;

    --I_opts)
        echo ${I_opts}
        ;;

    --L_opts)
        echo ${L_opts}
        ;;

    --R_opts)
        echo ${R_opts}
        ;;

    --ldflags)
        echo ${ldflags} ${L_opts} ${R_opts} ${libs}
        ;;

    --static)
        R_opts=""
        ;;

    *)
        usage
        exit 1
        ;;
    esac
    shift
done

exit 0
