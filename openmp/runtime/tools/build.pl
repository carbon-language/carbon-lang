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

# Standard modules.
use Data::Dumper;    # Not actually used, but useful for debugging dumps.

# Enable `libomp/tools/lib/' module directory.
use FindBin;
use lib "$FindBin::Bin/lib";

# LIBOMP modules.
use Build;
use LibOMP;
use Platform ":vars";
use Uname;
use tools;

our $VERSION = "0.017";

# --------------------------------------------------------------------------------------------------
# Important variables.
# --------------------------------------------------------------------------------------------------

my $root_dir  = $ENV{ LIBOMP_WORK };

my %makefiles = (
    rtl       => cat_file( $root_dir, "src",                       "makefile.mk" ),
    timelimit => cat_file( $root_dir, "tools", "src", "timelimit", "makefile.mk" ),
);

# --------------------------------------------------------------------------------------------------
# Parse command line.
# --------------------------------------------------------------------------------------------------

# Possible options.
#     * targets: comma separated list of targets the option has meaning for. For example,
#         "version" option (4 or 5) has a meaning only for "rtl" target, while "mode" option has
#         meaning for all targets.
#     * base: If base is true this is a base option. All the possible values of base options are
#         iterated if "--all" option is specified. If base is 0, this is an extra option.
#     * params: A hash of possible option values. "*" denotes default option value. For example,
#         if "versio" option is not specified, "--version=5" will be used implicitly.
#     * suffux: Only for extra options. Subroutine returning suffix for build and output
#         directories.
my $opts = {
    "target"          => { targets => "",                  base => 1, parms => { map( ( $_ => "" ), keys( %makefiles ) ), rtl => "*" }, },
    "version"         => { targets => "rtl",               base => 1, parms => { 5       => "*", 4         => ""              }, },
    "lib-type"        => { targets => "rtl",               base => 1, parms => { normal  => "*", stubs => ""                  }, },
    "link-type"       => { targets => "rtl",               base => 1, parms => { dynamic => "*", static    => ""              }, },
    "target-compiler" => { targets => "rtl,dsl",           base => 0, parms => { 12      => "*", 11        => ""              }, suffix => sub { $_[ 0 ];                       } },
    "mode"            => { targets => "rtl,dsl,timelimit", base => 0, parms => { release => "*", diag      => "", debug => "" }, suffix => sub { substr( $_[ 0 ], 0, 3 );       } },
    "omp-version"     => { targets => "rtl",               base => 0, parms => { 40 => "*", 30 => "", 25 => "" }, suffix => sub { $_[ 0 ]; } },
    "coverage"        => { targets => "rtl",               base => 0, parms => { off     => "*", on        => ""              }, suffix => sub { $_[ 0 ] eq "on" ? "c1" : "c0"; } },
    "tcheck"          => { targets => "rtl",               base => 0, parms => { 0       => "*", 1         => "", 2 => ""     }, suffix => sub { "t" . $_[ 0 ];                 } },
    "mic-arch"        => { targets => "rtl",               base => 0, parms => { knf     => "*", knc       => "", knl => ""   }, suffix => sub { $_[ 0 ];                       } },
    "mic-os"          => { targets => "rtl",               base => 0, parms => { bsd     => "*", lin       => ""              }, suffix => sub { $_[ 0 ];                       } },
    "mic-comp"        => { targets => "rtl",               base => 0, parms => { native  => "*", offload   => ""              }, suffix => sub { substr( $_[ 0 ], 0, 3 );       } },
};
my $synonyms = {
    "debug" => [ qw{ dbg debg } ],
};
# This array specifies order of options to process, so it cannot be initialized with keys( %$opts ).
my @all_opts   = qw{ target version lib-type link-type target-compiler mode omp-version coverage tcheck mic-arch mic-os mic-comp };
# This is the list of base options.
my @base_opts  = grep( $opts->{ $_ }->{ base } == 1, @all_opts );
# This is the list of extra options.
my @extra_opts = grep( $opts->{ $_ }->{ base } == 0, @all_opts );

sub suffix($$$) {
    my ( $opt, $value, $skip_if_default ) = @_;
    my $suffix = "";
    if ( not $skip_if_default or $value ne $opts->{ $opt }->{ dflt } ) {
        $suffix = $opts->{ $opt }->{ suffix }->( $value );
    }; # if
    return $suffix;
}; # sub suffix

my $scuts = {};     # Shortcuts. Will help to locate proper item in $opts.
foreach my $opt ( keys( %$opts ) ) {
    foreach my $parm ( keys( %{ $opts->{ $opt }->{ parms } } ) ) {
        if ( $parm !~ m{\A(?:[012]|on|off)\z} ) {
            $scuts->{ $parm } = $opts->{ $opt };
        }; # if
        if ( $opts->{ $opt }->{ parms }->{ $parm } eq "*" ) {
            $opts->{ $opt }->{ dflt } = $parm;
        }; # if
    }; # foreach $parm
}; # foreach $opt

sub parse_option(@) {
    # This function is called to process every option. $name is option name, $value is option value.
    # For boolean options $value is either 1 or 0,
    my ( $name, $value ) = @_;
    if ( $name eq "all" or $name eq "ALL" ) {
        foreach my $opt ( keys( %$opts ) ) {
            if ( $opts->{ $opt }->{ base } or $name eq "ALL" ) {
                foreach my $parm ( keys( %{ $opts->{ $opt }->{ parms } } ) ) {
                    $opts->{ $opt }->{ parms }->{ $parm } = 1;
                }; # foreach $parm
            }; # if
        }; # foreach $opt
        return;
    }; # if
    if ( exists( $opts->{ $name } ) ) {
        # Suppose it is option with explicit value, like "target=normal".
        if ( $value eq "all" ) {
            foreach my $parm ( keys( %{ $opts->{ $name }->{ parms } } ) ) {
                $opts->{ $name }->{ parms }->{ $parm } = 1;
            }; # foreach
            return;
        } elsif ( exists( $opts->{ $name }->{ parms }->{ $value } ) ) {
            $opts->{ $name }->{ parms }->{ $value } = 1;
            return;
        } elsif ( $value eq "" and exists( $opts->{ $name }->{ parms }->{ on } ) ) {
            $opts->{ $name }->{ parms }->{ on } = 1;
            return;
        } else {
            cmdline_error( "Illegal value of \"$name\" option: \"$value\"" );
        }; # if
    }; # if
    # Ok, it is not an option with explicit value. Try to treat is as a boolean option.
    if ( exists( $scuts->{ $name } ) ) {
        ( $value eq "1" or $value eq "0" ) or die "Internal error; stopped";
        $scuts->{ $name }->{ parms }->{ $name } = $value;
        return;
    }; # if
    # No, it is not a valid option at all.
    cmdline_error( "Illegal option: \"$name\"" );
}; # sub parse_option

my $clean        = 0;
my $clean_common = 0;
my $clobber      = 0;
my $test_deps    = 1;
my $test_touch   = 1;
my @goals;

sub synonyms($) {
    my ( $opt ) = @_;
    return exists( $synonyms->{ $opt } ) ? "|" . join( "|", @{ $synonyms->{ $opt } } ) : "";
}; # sub synonyms

my @specs = (
    map( ( "$_" . synonyms( $_ ) . "=s" => \&parse_option ), keys( %$opts  ) ),
    map( ( "$_" . synonyms( $_ ) . "!"  => \&parse_option ), keys( %$scuts ) ),
);
my $answer;
get_options(
    @specs,
    Platform::target_options(),
    "all"           => \&parse_option,
    "ALL"           => \&parse_option,
    "answer=s"      => \$answer,
    "test-deps!"    => \$test_deps,
    "test-touch!"   => \$test_touch,
    "version|ver:s" =>
        sub {
            # It is a tricky option. It specifies library version to build and it is also a standard
            # option to request tool version.
            if ( $_[ 1 ] eq "" ) {
                # No arguments => version request.
                print( "$tool version $VERSION\n" );
                exit( 0 );
            } else {
                # Arguments => version to build.
                parse_option( @_ )
            };
        },
);
@goals = @ARGV;
if ( grep( $_ eq "clobber", @goals ) ) {
    $clobber = 1;
}; # if
if ( grep( $_ eq "clean", @goals ) ) {
    $clean = 1;
}; # if

# Ok, now $opts is fulfilled with 0, 1 (explicitly set by the user) and "" and "*" (original
# values). In each option at least one 1 should be present (otherwise there is nothing to build).
foreach my $opt ( keys( %$opts ) ) {
    if ( not grep( $_ eq "1", values( %{ $opts->{ $opt }->{ parms } } ) ) ) {
        # No explicit "1" found. Enable default choice by replacing "*" with "1".
        foreach my $parm ( keys( %{ $opts->{ $opt }->{ parms } } ) ) {
            if ( $opts->{ $opt }->{ parms }->{ $parm } eq "*" ) {
                $opts->{ $opt }->{ parms }->{ $parm } = 1;
            }; # if
        }; # foreach $parm
    }; # if
}; # foreach $opt

# Clear $opts. Leave only "1".
foreach my $opt ( keys( %$opts ) ) {
    foreach my $parm ( keys( %{ $opts->{ $opt }->{ parms } } ) ) {
        if ( $opts->{ $opt }->{ parms }->{ $parm } ne "1" ) {
            delete( $opts->{ $opt }->{ parms }->{ $parm } );
        }; # if
    }; # foreach $parm
}; # foreach $opt

# --------------------------------------------------------------------------------------------------
# Fill job queue.
# --------------------------------------------------------------------------------------------------

sub enqueue_jobs($$@);
sub enqueue_jobs($$@) {
    my ( $jobs, $set, @rest ) = @_;
    if ( @rest ) {
        my $opt = shift( @rest );
        if (
            exists( $set->{ target } )
            and
            $opts->{ $opt }->{ targets } !~ m{(?:\A|,)$set->{ target }(?:,|\z)}
        ) {
            # This option does not have meananing for the target,
            # do not iterate, just use default value.
            enqueue_jobs( $jobs, { $opt => $opts->{ $opt }->{ dflt }, %$set }, @rest );
        } else {
            foreach my $parm ( sort( keys( %{ $opts->{ $opt }->{ parms } } ) ) ) {
                enqueue_jobs( $jobs, { $opt => $parm, %$set }, @rest );
            }; # foreach $parm
        }; # if
    } else {
        my $makefile  = $makefiles{ $set->{ target } };
        my @base      = map( substr( $set->{ $_ }, 0, 3 ), @base_opts );
        my @extra     = map( suffix( $_, $set->{ $_ }, 0 ), @extra_opts );
        my @ex        = grep( $_ ne "", map( suffix( $_, $set->{ $_ }, 1 ), @extra_opts ) );
            # Shortened version of @extra -- only non-default values.
        my $suffix    = ( @extra ? "." . join( ".", @extra ) : "" );
        my $knights   = index( $suffix, "kn" ) - 1;
        if ( $target_platform !~ "lrb" and $knights > 0 ) {
            $suffix = substr( $suffix, 0, $knights );
        }
        my $suf       = ( @ex ? "." . join( ".", @ex ) : "" );
            # Shortened version of $siffix -- only non-default values.
        my $build_dir = join( "-", $target_platform, join( "_", @base ) . $suffix, Uname::host_name() );
        my $out_arch_dir = cat_dir( $ENV{ LIBOMP_EXPORTS }, $target_platform . $suf );
        my $out_cmn_dir  = cat_dir( $ENV{ LIBOMP_EXPORTS }, "common" );
        push(
            @$jobs,
            {
                makefile => $makefile,
                make_args => [
                    "os="   . $target_os,
                    "arch=" . $target_arch,
                    "MIC_OS=" . $set->{ "mic-os" },
                    "MIC_ARCH=" . $set->{ "mic-arch" },
                    "MIC_COMP=" . $set->{ "mic-comp" },
                    "date=" . Build::tstr( $Build::start ),
                    "TEST_DEPS=" . ( $test_deps   ? "on" : "off" ),
                    "TEST_TOUCH=" . ( $test_touch ? "on" : "off" ),
                    "CPLUSPLUS=on",
                    "COVERAGE=" . $set->{ coverage },
                    # Option "mode" controls 3 make flags:
                    #     debug   => Full debugging   :    diagnostics,    debug info, no optimization.
                    #     diag    => Only diagnostics :    diagnostics,    debug info,    optimization.
                    #     release => Production build : no diagnostics, no debug info,    optimization.
                    "DEBUG_INFO=" .   ( $set->{ mode } ne "release" ? "on" : "off" ),
                    "DIAG=" .         ( $set->{ mode } ne "release" ? "on" : "off" ),
                    "OPTIMIZATION=" . ( $set->{ mode } ne "debug"   ? "on" : "off" ),
                    "LIB_TYPE=" . substr( $set->{ "lib-type" }, 0, 4 ),
                    "LINK_TYPE=" . substr( $set->{ "link-type" }, 0, 4 ),
                    "OMP_VERSION=" . $set->{ "omp-version" },
                    "USE_TCHECK=" . $set->{ tcheck },
                    "VERSION=" . $set->{ version },
                    "TARGET_COMPILER=" . $set->{ "target-compiler" },
                    "suffix=" . $suf,
                    @goals,
                ],
                build_dir  => $build_dir
            }
        ); # push
    }; # if
}; # sub enqueue_jobs

my @jobs;
enqueue_jobs( \@jobs, {}, @all_opts );

# --------------------------------------------------------------------------------------------------
# Do the work.
# --------------------------------------------------------------------------------------------------

my $exit = 0;

Build::init();

if ( $clobber ) {
    my @dirs = ( $ENV{ LIBOMP_TMP }, $ENV{ LIBOMP_EXPORTS }, cat_dir( $root_dir, "tools", "bin"  ) );
    my $rc = 0;
    question(
        "Clobber " . join( ", ", map( "\"" . Build::shorter( $_ ) . "\"", @dirs ) ) . " dirs? ",
        $answer,
        qr{\A(y|yes|n|no)\z}i
    );
    if ( $answer =~ m{\Ay}i ) {
        info( "Clobbering..." );
        $rc = Build::clean( @dirs );
        info( Build::rstr( $rc ) );
    }; # if
    if ( $rc != 0 ) {
        $exit = 3;
    }; # if
} else { # Build or clean.
    if ( @jobs ) {
        my $total = @jobs;    # Total number of jobs.
        my $n     = 0;        # Current job number.
        Build::progress( "", "" );     # Output empty line to log file.
        my $goals = join( " ", @goals );
        Build::progress( "Goals", $goals eq "" ? "(all)" : $goals );
        Build::progress( "Configurations", scalar( @jobs ) );
        foreach my $job ( @jobs ) {
            ++ $n;
            my $base = get_file( $job->{ build_dir } );
            Build::progress( "Making", "%3d of %3d : %s", $n, $total, $base );
            $job->{ rc } = Build::make( $job, $clean, sprintf( "%d/%d", $n, $total ) );
        }; # my $job
        my $failures = Build::summary();
        if ( $failures > 0 ) {
            $exit = 3;
        }; # if
    } else {
        info( "Nothing to do." );
    }; # if
}; # if

# And exit.
exit( $exit );

__END__

=pod

=head1 NAME


B<build.pl> -- Build one or more configurations of OMP RTL libraries.

=head1 SYNOPSIS

B<build.pl> I<option>... [B<-->] I<make-option>... I<variable>... I<goal>...

=head1 OPTIONS

=over

=item B<--all>

Build all base configurations.

=item B<--ALL>

Build really all configurations, including extra ones.

=item B<--answer=>I<str>

Use specified string as default answer to all questions.

=item B<--architecture=>I<arch>

Specify target architecture to build. Default is architecture of host machine. I<arch> can be C<32>,
C<32e>, or one of known aliases like C<IA32>.

If architecture is not specified explicitly, value of LIBOMP_ARCH environment variable is used.
If LIBOMP_ARCH is not defined, host architecture detected.

=item B<--os=>I<os>

Specify target OS. Default is OS of host machine. I<os> can be C<lin>, C<lrb>, C<mac>, C<win>,
or one of known aliases like C<Linux>, C<WinNT>, etc.

=item B<--mic-os=>I<os>

Specify OS on Intel(R) Many Integrated Core Architecture card. Default is C<bsd>. I<os> can be C<bsd>, C<lin>.

=item B<--mic-arch=>I<arch>

Specify architecture of Intel(R) Many Integrated Core Architecture card. Default is C<knf>. I<arch> can be C<knf>, C<knc>, C<knl>.

=item B<--mic-comp=>I<compiler-type>

Specify whether the Intel(R) Many Integrated Core Compiler is native or offload. Default is C<native>.
I<compiler-type> can be C<native> or C<offload>.

=item B<-->[B<no->]B<test-deps>

Enable or disable C<test-deps>. The test runs in any case, but result of disabled test is ignored.
By default, test is enabled.

=item B<-->[B<no->]B<test-touch>

Enable or disable C<test-touch>. The test runs in any case, but result of disabled test is ignored.
By default, test is enabled.

=item Base Configuration Selection Options

=over

=item B<--target=>I<target>

Build specified target, either C<rtl> (OMP Runtime Library; default),
or C<timelimit> (program used in testing), or C<all>.

=item B<--lib-type=>I<lib>

Build specified library, either C<normal> (default), or C<stubs>, or C<all>.

=item B<--link-type=>I<type>

Build specified link type, either C<dynamic> (default) or C<all>.

=back

=item Extra Configuration Selection Options

=over

=item B<--cover=>I<switch>

Build for code coverage data collection. I<switch> can be C<off> (default), C<on>
or C<all>.

=item B<--mode=>I<mode>

Build library of specified I<mode>, either C<debug>, C<diag>, C<release> (default), or C<all>.
Mode controls 3 features:

    ---------------------------------------------------
    feature/mode                   debug  diag  release
    ---------------------------------------------------
    debug info                       o      o
    diagnostics (asserts, traces)    o      o
    code optimization                       o      o
    ---------------------------------------------------

=item B<--target-compiler=>I<version>

Build files for specified target compiler, C<11> or C<12>.

=back

=item Shortcuts

If option with C<no> prefix is used, corresponding configuration will B<not> be built.
Useful for excluding some configurations if one or more other options specified with C<all>
value (see Examples).

=over

=item B<-->[B<no>]B<11>

Build files for compiler C<11>.

=item B<-->[B<no>]B<12>

Build files for compiler C<12>.

=item B<-->[B<no>]B<debug>

=item B<-->[B<no>]B<debg>

=item B<-->[B<no>]B<dbg>

Build debuggable library.

=item B<-->[B<no>]B<diag>

Build library with diagnostics enabled.

=item B<-->[B<no>]B<dynamic>

Build dynamic library (default).

=item B<-->[B<no>]B<normal>

Build normal library (default).

=item B<-->[B<no>]B<release>

Build release library (default).

=item B<-->[B<no>]B<rtl>

Build OMP RTL (default).

=item B<-->[B<no>]B<stubs>

Build stubs library.

=item B<-->[B<no>]B<timelimit>

Build timelimit utility program.

=back

=item Standard Options

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

Print program version and exit.

=item B<--quiet>

Work quiet, do not print informational messages.

=back

=back

=head1 ARGUMENTS

=over

=item I<make-option>

Any option for makefile, for example C<-k> or C<-n>. If you pass some options to makefile, C<-->
delimiter is mandatory, otherwise C<build.pl> processes all the options internally.

=item I<variable>

Define makefile variable in form I<name>B<=>I<value>. Most makefile capabilities are
accessible through C<build.pl> options, so there is no need in defining make variables in command
line.

=item I<goal>

Makefile goal to build (or clean).

=over

=item B<all>

Build C<lib>, C<tests>, C<inc>.

=item B<common>

Build common (architecture-independent) files. Common files are not configuration-dependent, so
there is no point in building it for more than one configuration (thought it is harmless).
However, do not build common files on many machines simultaneously.

=item B<clean>

Delete the export files and clean build directory of configuration(s) specified by options. Note
that C<clean> goal cannot be mixed with other goals (except for C<clean-common>).

=item B<clean-common>

Delete the common files in F<exports/> directory.

=item B<clobber>

Clean F<export/> and F<tmp/> directories. If C<clobber> is specified, other goals and/or options
do not matter.

Note: Clobbering is potentialy dangerous operation, because it deletes content of directory
pointed by If C<LIBOMP_TMP> environment variable, so C<build.pl> asks a confirmation before
clobbering. To suppress the question, use option C<--answer=yes>.

=item B<fat>

C<mac_32e> only: Build fat libraries for both mac_32 and mac_32e. Should be run when C<lib>
goal is built on both C<mac_32> and C<mac_32e>.

=item I<file.o>

(Windows* OS: I<file.obj>) Build specified object file only.

=item I<file.i>

Create preprocessed source file.

=item B<force-tests>

Force performing tests.

=item B<force-test-deps>

Force performing test-deps.

=item B<force-test-instr>

Force performing test-instr.

=item B<force-test-relo>

Force performing test-relo.

=item B<force-test-touch>

Force performing test-touch.

=item B<inc>

Build Fortran include files, omp_lib.h, omp_lib.mod and omp_lib_kinds.mod.

=item B<lib>

Build library (on Windows* OS in case of dynamic linking, it also builds import library).

=item B<tests>

Perform tests: C<test-deps>, C<test-instr>, C<test-relo>, and C<test-touch>.

=item B<test-deps>

Check the library dependencies. 

=item B<test-instr>

Intel(R) Many Integrated Core Architecture only: check the library does not contain undesired instructions.

=item B<test-relo>

Linux* OS with dynamic linking only: check the library does not contain position-dependent
code.

=item B<test-touch>

Build a very simple application with native compiler (GNU on Linux* OS and OS X*, MS
on Windows* OS), check it does not depend on C<libirc> library, and run it.

=back

=back

=head1 DESCRIPTION

C<build.pl> constructs the name of a build directory, creates the directory if it
does not exist, changes to it, and runs make to build the goals in specified configuration.
If more than one configuration are specified in command line C<build.pl> builds them all.

Being run with C<clean> goal, C<build.pl> does not build but deletes export files and
cleans build directories of configuration specified by other options. For example,
C<build.pl --all clean> means "clean build directories for all configurations",
it does B<not> mean "clean then build all".

C<clear-common> goal deletes common files in F<exports/> directory.
Since common files are really common and not architecture and/or configuration dependent,
there are no much meaning in combining C<clear-common> with configuration selection options.
For example, C<build.pl --all clean-common> deletes the same files 13 times.
However, it does not hurt and can be used in conjunction with C<clear> goal.

C<clobber> goal instructs C<build.pl> to clean exports and all build
directories, e. g. clean everything under F<exports/> and F<tmp/> directories.

Logs are saved automatically, there is no need in explicit output redirection.
Log file for each particular configuration is named F<build.log> and located in build directory.
Summary log file (just result of each configuration) is saved in F<tmp/> directory.

Log files are never overwritten. C<build.pl> always appends output to log files.
However (obviously), C<clear> deletes log file for cleared configurations,
and C<clobber> deletes all summary log files.

=head2 Environment Variables

=over

=item B<LIBOMP_ARCH>

Specifies target architecture. If not present, host architecture is used. Environment variable may
be overriden by C<--architecture> command line option.

=item B<LIBOMP_EXPORTS>

Specifies directory for output files. If not set, C<$LIBOMP_WORK/exports/> used by default.

=item B<LIBOMP_OS>

Specifies target OS. If not present, host OS is used. Environment variable may
be overriden by C<--os> command line option.

=item B<LIBOMP_TMP>

Directory for temporary files. C<build.pl> creates build directories there. If not set,
C<$LIBOMP_WORK/tmp/> used by default.

On Windows* OS F<tmp/> directory on local drive speeds up the build process.

=item B<LIBOMP_WORK>

Root of libomp directory tree, contains F<src/>, F<tools/>, and F<exports/> subdirs.
If not set, C<build.pl> guesses the root dir (it is a parent of dir containing C<build.pl>).

Note: Guessing it not reliable. Please set C<LIBOMP_WORK> environment variable appropriately.

=back

=head1 EXAMPLES

=head2 Development

Build normal (performance) dynamic library for debugging:

    $ build.pl --debug

Build all libraries (normal, stub; dynamic RTL) for debugging:

    $ build.pl --all --debug

Do a clean build for all:

    $ build.pl --all --debug clean && build.pl --all --debug

Debugging libraries are saved in F<exports/I<platform>.deb/>.

=head2 Promotion

=over

=item 1

Clobber everything; on one machine:

    $ build.pl clobber

=item 2

Build common headers, on one machine:

    $ build.pl common

=item 3

Build all platform-dependent files, on all machines:

    $ build.pl --all

=item 4

Build OS X* universal (fat) libraries, on C<mac_32e>:

    $ build.pl fat

=back

=cut

# end of file #
