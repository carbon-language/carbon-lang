#!/usr/bin/perl

#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# Pragmas.
use strict;
use warnings;

use FindBin;
use lib "$FindBin::Bin/lib";

# LIBOMP modules.
use Platform ":vars";
use tools;

our $VERSION = "0.015";

my $pedantic;

# --------------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------------------


sub run($\$\$;\$) {
    my ( $cmd, $stdout, $stderr, $path ) = @_;
    my ( @path, $rc );
    @path = which( $cmd->[ 0 ], -all => 1 );
    if ( @path > 0 ) {
        if ( @path > 1 and $pedantic ) {
            warning( "More than one \"$cmd->[ 0 ]\" found in PATH:", map( "    $_", @path ) );
        }; # if
        debug( "\"$cmd->[ 0 ]\" full path is \"$path[ 0 ]\"." );
        if ( defined( $path ) ) {
            $$path = $path[ 0 ];
        }; # if
        debug( "Executing command: \""  . join ( " ", @$cmd ) . "\"." );
        $rc =
            execute(
                $cmd,
                -ignore_signal => 1, -ignore_status => 1,
                -stdout => $stdout, -stderr => $stderr, -stdin => undef
            );
        if ( $rc < 0 ) {
            warning( "Cannot run \"$cmd->[ 0 ]\": $@" );
        }; # if
        debug( "stdout:", $$stdout, "(eof)", "stderr:", $$stderr, "(eof)" );
    } else {
        warning( "No \"$cmd->[ 0 ]\" found in PATH." );
        $rc = -1;
    }; # if
    return $rc;
}; # sub run


sub get_arch($$$) {
    my ( $name, $str, $exps ) = @_;
    my ( $arch, $count );
    $count = 0;
    foreach my $re ( keys( %$exps ) ) {
        if ( $str =~ $re ) {
            $arch = $exps->{ $re };
            ++ $count;
        }; # if
    }; # for
    if ( $count != 1 or not Platform::canon_arch( $arch ) ) {
        warning( "Cannot detect $name architecture: $str" );
        return undef;
    }; # if
    return $arch;
}; # sub get_arch

sub encode($) {
    my ( $str ) = @_;
    $str =~ s{ }{_}g;
    return $str;
}; # sub encode


# --------------------------------------------------------------------------------------------------
# get_xxx_version subroutines.
# --------------------------------------------------------------------------------------------------
#
# Some of get_xxx_version() subroutines accept an argument -- a tool name. For example,
# get_intel_compiler_version() can report version of C, C++, or Fortran compiler. The tool for
# report should be specified by argument, for example: get_intel_compiler_version( "ifort" ).
#
# get_xxx_version() subroutines returns list of one or two elements:
#     1. The first element is short tool name (like "gcc", "g++", "icl", etc).
#     2. The second element is version string.
# If returned list contain just one element, it means there is a problem with the tool.
#

sub get_perl_version() {
    my ( $rc, $stdout, $stderr, $version );
    my $tool = "perl";
    my ( @ret ) = ( $tool );
    $rc = run( [ $tool, "--version" ], $stdout, $stderr );
    if ( $rc >= 0 ) {
        # Typical perl output:
        #    This is perl, v5.10.0 built for x86_64-linux-thread-multi
        #    This is perl, v5.8.8 built for MSWin32-x64-multi-thread
        #    This is perl, v5.10.1 (*) built for x86_64-linux-thread-multi
        if ( $stdout !~ m{^This is perl.*v(\d+\.\d+(?:\.\d+)).*built for}m ) {
            warning( "Cannot parse perl output:", $stdout, "(oef)" );
        }; # if
        $version = $1;
        if ( $target_os eq "win" ) {
            if ( $stdout !~ m{Binary build (.*) provided by ActiveState } ) {
                warning( "Perl is not ActiveState one" );
            }; # if
        }; # if
    }; # if
    push( @ret, $version );
    return @ret;
}; # sub get_perl_version


sub get_gnu_make_version() {
    my ( $rc, $stdout, $stderr, $version );
    my $tool = "make";
    my ( @ret ) = ( $tool );
    my ( $path );
    $rc = run( [ $tool, "--version" ], $stdout, $stderr, $path );
    if ( $rc >= 0 ) {
        # Typical make output:
        #     GNU Make version 3.79.1, by Richard Stallman and Roland McGrath.
        #     GNU Make 3.81
        if ( $stdout =~ m{^GNU Make (?:version )?(\d+\.\d+(?:\.\d+)?)(?:,|\s)} ) {
            $version = $1;
        }; # if
        if ( $target_os eq "win" and $stdout =~ m{built for ([a-z0-9-]+)} ) {
            my $built_for = $1;
            debug( "GNU Make built for: \"$built_for\"." );
            if ( $built_for =~ m{cygwin}i ) {
                warning( "\"$path\" is a Cygwin make, it is *not* suitable." );
                return @ret;
            }; # if
        }; # if
    }; # if
    push( @ret, $version );
    return @ret;
}; # sub get_gnu_make_version


sub get_intel_compiler_version($) {
    my ( $tool ) = @_;    # Tool name, like "icc", "icpc", "icl", or "ifort".
    my ( @ret ) = ( $tool );
    my ( $rc, $stdout, $stderr, $tool_re );
    my $version;
    my $ic_archs = {
        qr{32-bit|IA-32}        => "32",
        qr{Intel\(R\) 64} => "32e",
        qr{Intel\(R\) [M][I][C] Architecture} => "32e",
    };
    $tool_re = quotemeta( $tool );
    $rc = run( [ $tool, ( $target_os eq "win" ? () : ( "-V" ) ) ], $stdout, $stderr );
    if ( $rc < 0 ) {
        return @ret;
    }; # if
    # Intel compiler version string is in the first line of stderr. Get it.
    #$stderr =~ m{\A(.*\n?)};
    # AC: Let's look for version string in the first line which contains "Intel" string.
    #     This allows to use 11.1 and 12.0 compilers on new MAC machines by ignoring
    #     huge number of warnings issued by old compilers.
    $stderr =~ m{^(Intel.*)$}m;
    my $vstr = $1;
    my ( $apl, $ver, $bld, $pkg );
    if ( 0 ) {
    } elsif ( $vstr =~ m{^Intel.*?Compiler\s+(.*?),?\s+Version\s+(.*?)\s+Build\s+(\S+)(?:\s+Package ID: (\S+))?} ) {
        # 9.x, 10.x, 11.0.
        ( $apl, $ver, $bld, $pkg ) = ( $1, $2, $3, $4 );
    } elsif ( $vstr =~ m{^Intel's (.*?) Compiler,?\s+Version\s+(.*?)\s+Build\s+(\S+)} ) {
        # 11.1
        ( $apl, $ver, $bld ) = ( $1, $2, $3 );
    } else {
        warning( "Cannot parse ${tool}'s stderr:", $stderr, "(eof)" );
        return @ret;
    }; # if
    my $ic_arch = get_arch( "Intel compiler", $apl, $ic_archs );
    if ( not defined( $ic_arch ) ) {
        return @ret;
    }; # if
    if ( Platform::canon_arch( $ic_arch ) ne $target_arch ) {
        warning( "Target architecture is $target_arch, $tool for $ic_arch found." );
        return @ret;
    }; # if
    # Normalize version.
    my $stage;
    $ver =~ s{\s+}{ }g;
    $ver = lc( $ver );
    if ( $ver =~ m{\A(\d+\.\d+(?:\.\d+)?) ([a-z]+)\a}i ) {
        ( $version, $stage ) = ( $1, $2 );
    } else {
        ( $version, $stage ) = ( $ver, "" );
    }; # if
    # Parse package.
    if ( defined( $pkg ) ) {
        if ( $pkg !~ m{\A[lwm]_[a-z]+_[a-z]_(\d+\.\d+\.\d+)\z}i ) {
            warning( "Cannot parse Intel compiler package: $pkg" );
            return @ret;
        }; # if
        $pkg = $1;
        $version = $pkg;
    }; # if
    push( @ret, "$version " . ( $stage ? "$stage " : "" ) . "($bld) for $ic_arch" );
    # Ok, version of Intel compiler found successfully. Now look at config file.
    # Installer of Intel compiler tends to add a path to MS linker into compiler config file.
    # It leads to troubles. For example, all the environment set up for MS VS 2005, but Intel
    # compiler uses lnker from MS VS 2003 because it is specified in config file.
    # To avoid such troubles, make sure:
    #     ICLCFG/IFORTCFG environment variable exists or
    #     compiler config file does not exist, or
    #     compiler config file does not specify linker.
    if ( $target_os eq "win" ) {
        if ( not exists( $ENV{ uc( $tool . "cfg" ) } ) ) {
            # If ICLCFG/IFORTCFG environment varianle exists, everything is ok.
            # Otherwise check compiler's config file.
            my $path = which( $tool );
            $path =~ s{\.exe\z}{}i;     # Drop ".exe" suffix.
            $path .= ".cfg";            # And add ".cfg" one.
            if ( -f $path ) {
                # If no config file exists, it is ok.
                # Otherwise analyze its content.
                my $bulk = read_file( $path );
                $bulk =~ s{#.*\n}{}g;    # Remove comments.
                my @options = ( "Qvc", "Qlocation,link," );
                foreach my  $opt ( @options ) {
                    if ( $bulk =~ m{[-/]$opt} ) {
                        warning( "Compiler config file \"$path\" contains \"-$opt\" option." );
                    }; # if
                }; # foreach
            }; # if
        }; # if
    }; # if
    return @ret;
}; # sub get_intel_compiler_version


sub get_gnu_compiler_version($) {
    my ( $tool ) = @_;
    my ( @ret ) = ( $tool );
    my ( $rc, $stdout, $stderr, $version );
    $rc = run( [ $tool, "--version" ], $stdout, $stderr );
    if ( $rc >= 0 ) {
        my ( $ver, $bld );
        if ( $target_os eq "mac" ) {
            # i686-apple-darwin8-gcc-4.0.1 (GCC) 4.0.1 (Apple Computer, Inc. build 5367)
            # i686-apple-darwin9-gcc-4.0.1 (GCC) 4.0.1 (Apple Inc. build 5484)
            # i686-apple-darwin11-llvm-gcc-4.2 (GCC) 4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2336.9.00)
            $stdout =~ m{^.*? \(GCC\) (\d+\.\d+\.\d+) \(.*Apple.*?Inc\. build (\d+)\)}m;
            ( $ver, $bld ) = ( $1, $2 );
        } else {
            if ( 0 ) {
            } elsif ( $stdout =~ m{^.*? \(GCC\) (\d+\.\d+\.\d+)(?: (\d+))?}m ) {
                # g++ (GCC) 3.2.3 20030502 (Red Hat Linux 3.2.3-20)
                # GNU Fortran (GCC) 4.3.2 20081105 (Red Hat 4.3.2-7)
                ( $ver, $bld ) = ( $1, $2 );
            } elsif ( $stdout =~ m{^.*? \(SUSE Linux\) (\d+\.\d+\.\d+)\s+\[.*? (\d+)\]}m ) {
                # gcc (SUSE Linux) 4.3.2 [gcc-4_3-branch revision 141291]
                ( $ver, $bld ) = ( $1, $2 );
            } elsif ( $stdout =~ m{^.*? \(SUSE Linux\) (\d+\.\d+\.\d+)\s+\d+\s+\[.*? (\d+)\]}m ) {
                # gcc (SUSE Linux) 4.7.2 20130108 [gcc-4_7-branch revision 195012]
                ( $ver, $bld ) = ( $1, $2 );
            } elsif ( $stdout =~ m{^.*? \((Debian|Ubuntu).*?\) (\d+\.\d+\.\d+)}m ) {
                # gcc (Debian 4.7.2-22) 4.7.2
                # Debian support from Sylvestre Ledru 
                # Thanks!
                $ver = $2;
            }; # if
        }; # if
        if ( defined( $ver ) ) {
            $version = $ver . ( defined( $bld ) ? " ($bld)" : "" );
        } else {
            warning( "Cannot parse GNU compiler version:", $stdout, "(eof)" );
        }; # if
    }; # if
    push( @ret, $version );
    return @ret;
}; # sub get_gnu_compiler_version


sub get_clang_compiler_version($) {
    my ( $tool ) = @_;
    my ( @ret ) = ( $tool );
    my ( $rc, $stdout, $stderr, $version );
    $rc = run( [ $tool, "--version" ], $stdout, $stderr );
    if ( $rc >= 0 ) {
        my ( $ver, $bld );
        if ( $target_os eq "mac" ) {
            # Apple LLVM version 4.2 (clang-425.0.28) (based on LLVM 3.2svn)
            $stdout =~ m{^.*? (\d+\.\d+) \(.*-(\d+\.\d+\.\d+)\)}m;
            ( $ver, $bld ) = ( $1, $2 );
            # For custom clang versions.
            if ( not defined($ver) and $stdout =~ m{^.*? (\d+\.\d+)( \((.*)\))?}m ) {
                ( $ver, $bld ) = ( $1, $3 );
            }
        } else {
            if ( 0 ) {
            } elsif ( $stdout =~ m{^.*? (\d+\.\d+)( \((.*)\))?}m ) {
                # clang version 3.3 (tags/RELEASE_33/final)
                ( $ver, $bld ) = ( $1, $3 );
            } 
        }; # if
        if ( defined( $ver ) ) {
            $version = $ver . ( defined( $bld ) ? " ($bld)" : "" );
        } else {
            warning( "Cannot parse Clang compiler version:", $stdout, "(eof)" );
        }; # if
    }; # if
    push( @ret, $version );
    return @ret;
}; # sub get_gnu_compiler_version


sub get_ms_compiler_version() {
    my ( $rc, $stdout, $stderr, $version );
    my $tool = "cl";
    my ( @ret ) = ( $tool );
    my $mc_archs = {
        qr{80x86|x86}     => "IA-32 architecture",
        qr{AMD64|x64}     => "Intel(R) 64",
    };
    $rc = run( [ $tool ], $stdout, $stderr );
    if ( $rc < 0 ) {
        return @ret;
    }; # if
    if ( $stderr !~ m{^Microsoft .* Compiler Version (.*?) for (.*)\s*$}m ) {
        warning( "Cannot parse MS compiler output:", $stderr, "(eof)" );
        return @ret;
    }; # if
    my ( $ver, $apl ) = ( $1, $2 );
    if ( $ver !~ m{\A\d+(?:\.\d+)+\z} ) {
        warning( "Cannot parse MS compiler version: $ver" );
        return @ret;
    }; # if
    my $mc_arch = get_arch( "MS compiler", $apl, $mc_archs );
    if ( not defined( $mc_arch ) ) {
        return @ret;
    }; # if
    if ( Platform::canon_arch( $mc_arch ) ne $target_arch ) {
        warning( "Target architecture is $target_arch, $tool for $mc_arch found" );
        return @ret;
    }; # if
    $version = "$ver for $target_arch";
    push( @ret, $version );
    return @ret;
}; # sub get_ms_compiler_version


sub get_ms_linker_version() {
    my ( $rc, $stdout, $stderr, $version );
    my $tool = "link";
    my ( @ret ) = ( $tool );
    my ( $path );
    $rc = run( [ $tool ], $stdout, $stderr, $path );
    if ( $rc < 0 ) {
        return @ret;
    }; # if
    if ( $stdout !~ m{^Microsoft \(R\) Incremental Linker Version (\d+(?:\.\d+)+)\s*$}m ) {
        warning( "Cannot parse MS linker output:", $stdout, "(eof)" );
        if ( $stderr =~ m{^link: missing operand} ) {
            warning( "Seems \"$path\" is a Unix-like \"link\" program, not MS linker." );
        }; # if
        return @ret;
    }; # if
    $version = ( $1 );
    push( @ret, $version );
    return @ret;
}; # sub get_ms_linker_version


# --------------------------------------------------------------------------------------------------
# "main" program.
# --------------------------------------------------------------------------------------------------

my $make;
my $intel       = 1;             # Check Intel compilers.
my $fortran     = 0;             # Check for corresponding Fortran compiler, ifort for intel 
                                 #                                           gfortran for gnu 
                                 #                                           gfortran for clang 
my $clang       = 0;             # Check Clang Compilers.
my $intel_compilers = {
    "lin" => { c => "icc", cpp => "icpc", f => "ifort" },
    "lrb" => { c => "icc", cpp => "icpc", f => "ifort" },
    "mac" => { c => "icc", cpp => "icpc", f => "ifort" },
    "win" => { c => "icl", cpp => undef,  f => "ifort" },
};
my $gnu_compilers = {
    "lin" => { c => "gcc", cpp =>  "g++", f => "gfortran" },
    "mac" => { c => "gcc", cpp =>  "g++", f => "gfortran" },
};
my $clang_compilers = {
    "lin" => { c => "clang", cpp =>  "clang++" },
    "mac" => { c => "clang", cpp =>  "clang++" },
};

get_options(
    Platform::target_options(),
    "intel!"         => \$intel,
    "fortran"        => \$fortran,
    "clang"          => \$clang,
    "make"           => \$make,
    "pedantic"       => \$pedantic,
);

my @versions;
push( @versions, [ "Perl",     get_perl_version() ] );
push( @versions, [ "GNU Make", get_gnu_make_version() ] );
if ( $intel ) {
    my $ic = $intel_compilers->{ $target_os };
    push( @versions, [ "Intel C Compiler",       get_intel_compiler_version( $ic->{ c } ) ] );
    if ( defined( $ic->{ cpp } ) ) {
        # If Intel C++ compiler has a name different from C compiler, check it as well.
        push( @versions, [ "Intel C++ Compiler", get_intel_compiler_version( $ic->{ cpp } ) ] );
    }; # if
    # fortran check must be explicitly specified on command line with --fortran
    if ( $fortran ) {
        if ( defined( $ic->{ f } ) ) {
            push( @versions, [ "Intel Fortran Compiler", get_intel_compiler_version( $ic->{ f } ) ] );
        }; # if
    };
}; # if
if ( $target_os eq "lin" or $target_os eq "mac" ) {
    # check for clang/gnu tools because touch-test.c is compiled with them.
    if ( $clang or $target_os eq "mac" ) { # OS X* >= 10.9 discarded GNU compilers.
        push( @versions, [ "Clang C Compiler",     get_clang_compiler_version( $clang_compilers->{ $target_os }->{ c   } ) ] );
        push( @versions, [ "Clang C++ Compiler",   get_clang_compiler_version( $clang_compilers->{ $target_os }->{ cpp } ) ] );
    } else {
        push( @versions, [ "GNU C Compiler",     get_gnu_compiler_version( $gnu_compilers->{ $target_os }->{ c   } ) ] );
        push( @versions, [ "GNU C++ Compiler",   get_gnu_compiler_version( $gnu_compilers->{ $target_os }->{ cpp } ) ] );
    };
    # if intel fortran has been checked then gnu fortran is unnecessary
    # also, if user specifies clang as build compiler, then gfortran is assumed fortran compiler
    if ( $fortran and not $intel ) {
        push( @versions, [ "GNU Fortran Compiler", get_gnu_compiler_version( $gnu_compilers->{ $target_os }->{ f } ) ] );
    }; 
}; 
if ( $target_os eq "win" ) {
    push( @versions, [ "MS C/C++ Compiler",  get_ms_compiler_version() ] );
    push( @versions, [ "MS Linker",          get_ms_linker_version() ] );
}; # if

my $count = 0;
foreach my $item ( @versions ) {
    my ( $title, $tool, $version ) = @$item;
    if ( not defined( $version ) ) {
        $version = "--- N/A ---";
        ++ $count;
    }; # if
    if ( $make ) {
        printf( "%s=%s\n", encode( $tool ), encode( $version ) );
    } else {
        printf( "%-25s: %s\n", $title, $version );
    }; # if
}; # foreach

exit( $count == 0 ? 0 : 1 );

__END__

=pod

=head1 NAME

B<check-tools.pl> -- Check development tools availability and versions.

=head1 SYNOPSIS

B<check-tools.pl> I<OPTION>...

=head1 OPTIONS

=over

=item B<--make>

Produce output suitable for using in makefile: short tool names (e. g. "icc" instead of "Intel C
Compiler"), spaces in version strings replaced with underscores.

=item Tools selection

=over

=item B<-->[B<no->]B<-gnu-fortran>

Check GNU Fortran compiler. By default, it is not checked.

=item B<-->[B<no->]B<intel>

Check Intel C, C++ and Fortran compilers. This is default.

=back

=item Platform selection

=over

=item B<--architecture=>I<str>

Specify target architecture. Used in cross-builds, for example when building 32-bit applications on
Intel(R) 64 machine.

If architecture is not specified explicitly, value of LIBOMP_ARCH environment variable is used.
If LIBOMP_ARCH is not defined, host architecture detected.

=item B<--os=>I<str>

Specify target OS name. Used in cross-builds, for example when building Intel(R) Many Integrated Core Architecture applications on
Windows* OS.

If OS is not specified explicitly, value of LIBOMP_OS environment variable is used.
If LIBOMP_OS is not defined, host OS detected.

=back

=back

=head2 Standard Options

=over

=item B<--doc>

=item B<--manual>

Print full help message and exit.

=item B<--help>

Print short help message and exit.

=item B<--usage>

Print very short usage message and exit.

=item B<--verbose>

Do print informational messages.

=item B<--version>

Print version and exit.

=item B<--quiet>

Work quiet, do not print informational messages.

=back

=head1 DESCRIPTION

This script checks availability and versions of development tools. By default, the script checks:
Perl, GNU Make, Intel compilers, GNU C and C++ compilers (Linux* OS and OS X*),
Microsoft C/C++ compiler and linker (Windows* OS).

The sript prints nice looking table or machine-readable strings.

=head2 EXIT

=over

=item *

0 -- All programs found.

=item *

1 -- Some of tools are not found.

=back

=head1 EXAMPLES

    $ check-tools.pl
    Perl                     : 5.8.0
    GNU Make                 : 3.79.1
    Intel C Compiler         : 11.0 (20080930) for 32e
    Intel C++ Compiler       : 11.0 (20080930) for 32e
    Intel Fortran Compiler   : 10.1.008 (20070913) for 32e
    GNU C Compiler           : 3.2.3 (20030502)
    GNU C++ Compiler         : 3.2.3 (20030502)

    > check-tools.pl --make
    perl=5.8.8
    make=3.81
    icl=10.1_(20070913)_for_32e
    ifort=10.1_(20070913)_for_32e
    cl=14.00.40310.41_for_32e
    link=8.00.40310.39

=back

=cut

# end of file #

