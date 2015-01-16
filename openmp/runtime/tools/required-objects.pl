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

use strict;
use warnings;

use File::Glob ":glob";
use Data::Dumper;

use FindBin;
use lib "$FindBin::Bin/lib";

use tools;
use Platform ":vars";

our $VERSION = "0.004";

# --------------------------------------------------------------------------------------------------
# Set of objects:       # Ref to hash, keys are names of objects.
#     object0:          # Ref to hash of two elements with keys "defined" and "undefined".
#         defined:      # Ref to array of symbols defined in object0.
#             - symbol0 # Symbol name.
#             - ...
#         undefined:    # Ref to array of symbols referenced in object0.
#             - symbol0
#             - ...
#     object1:
#         ...
#     ...
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Set of symbols:       # Ref to hash, keys are names of symbols.
#    symbol0:           # Ref to array of object names where the symbol0 is defined.
#        - object0      # Object file name.
#        - ...
#    symbol1:
#        ...
#    ...
# --------------------------------------------------------------------------------------------------

sub dump_objects($$$) {

    my ( $title, $objects, $dump ) = @_;

    if ( $dump > 0 ) {
        STDERR->print( $title, "\n" );
        foreach my $object ( sort( keys( %$objects ) ) ) {
            STDERR->print( "    $object\n" );
            if ( $dump > 1 ) {
                STDERR->print( "        Defined symbols:\n" );
                foreach my $symbol ( sort( @{ $objects->{ $object }->{ defined } } ) ) {
                    STDERR->print( "            $symbol\n" );
                }; # foreach $symbol
                STDERR->print( "        Undefined symbols:\n" );
                foreach my $symbol ( sort( @{ $objects->{ $object }->{ undefined } } ) ) {
                    STDERR->print( "            $symbol\n" );
                }; # foreach $symbol
            }; # if
        }; # foreach $object
    }; # if

}; # sub dump_objects

sub dump_symbols($$$) {

    my ( $title, $symbols, $dump ) = @_;

    if ( $dump > 0 ) {
        STDERR->print( $title, "\n" );
        foreach my $symbol ( sort( keys( %$symbols ) ) ) {
            STDERR->print( "    $symbol\n" );
            if ( $dump > 1 ) {
                foreach my $object ( sort( @{ $symbols->{ $symbol } } ) ) {
                    STDERR->print( "        $object\n" );
                }; # foreach
            }; # if
        }; # foreach $object
    }; # if

}; # sub dump_symbols

# --------------------------------------------------------------------------------------------------
# Name:
#     load_symbols -- Fulfill objects data structure with symbol names.
# Synopsis:
#     load_symbols( $objects );
# Arguments:
#     $objects (in/out) -- Set of objects. On enter, it is expected that top-level hash has filled
#         with object names only. On exit, it is completely fulfilled with lists of symbols
#         defined or referenced in each object file.
# Returns:
#     Nothing.
# Example:
#     my $objects = { foo.o => {} };
#     load_symbols( $objects );
#     # Now $objects is { goo.o => { defined => [ ... ], undefined => [ ... ] } }.
#
# --------------------------------------------------------------------------------------------------
# This version of load_symbols parses output of nm command and works on Linux* OS and OS X*.
#
sub _load_symbols_nm($) {

    my $objects = shift( @_ );
        # It is a ref to hash. Keys are object names, values are empty hashes (for now).
    my @bulk;

    if ( %$objects ) {
        # Do not run nm if a set of objects is empty -- nm will try to open a.out in this case.
        my $tool;
        if($target_arch eq "mic") {
            $tool = "x86_64-k1om-linux-nm"
        } else {
            $tool = "nm"
        }
        execute(
            [
                $tool,
                "-g",    # Display only external (global) symbols.
                "-o",    # Precede each symbol by the name of the input file.
                keys( %$objects )
                    # Running nm once (rather than once per object) improves performance
                    # drastically.
            ],
            -stdout => \@bulk
        );
    }; # if

    foreach my $line ( @bulk ) {
        if ( $line !~ m{^(.*):(?: ?[0-9a-f]*| *) ([A-Za-z]) (.*)$} ) {
            die "Cannot parse nm output, line:\n    $line\n";
        }; # if
        my ( $file, $tag, $symbol ) = ( $1, $2, $3 );
        if ( not exists( $objects->{ $file } ) ) {
            die "nm reported unknown object file:\n    $line\n";
        }; # if
        # AC: exclude some libc symbols from renaming, otherwise we have problems
        #     in tests for gfortran + static libiomp on Lin_32.
        #     These symbols came from libtbbmalloc.a
        if ( $target_os eq "lin" ) {
            if ( $symbol =~ m{__i686} ) {
                next;
            }
        }
        # AC: added "w" to tags of undefined symbols, e.g. malloc is weak in libirc v12.1.
        if ( $tag eq "U" or $tag eq "w" ) { # Symbol not defined.
            push( @{ $objects->{ $file }->{ undefined } }, $symbol );
        } else {             # Symbol defined.
            push( @{ $objects->{ $file }->{ defined } }, $symbol );
        }; # if
    }; # foreach

    return undef;

}; # sub _load_symbols_nm

# --------------------------------------------------------------------------------------------------
# This version of load_symbols parses output of link command and works on Windows* OS.
#
sub _load_symbols_link($) {

    my $objects = shift( @_ );
        # It is a ref to hash. Keys are object names, values are empty hashes (for now).
    my @bulk;

    if ( %$objects ) {
        # Do not run nm if a set of objects is empty -- nm will try to open a.out in this case.
        execute(
            [
                "link",
                "/dump",
                "/symbols",
                keys( %$objects )
                    # Running nm once (rather than once per object) improves performance
                    # drastically.
            ],
            -stdout => \@bulk
        );
    }; # if

    my $num_re   = qr{[0-9A-F]{3,4}};
    my $addr_re  = qr{[0-9A-F]{8}};
    my $tag_re   = qr{DEBUG|ABS|UNDEF|SECT[0-9A-F]+};
    my $class_re = qr{Static|External|Filename|Label|BeginFunction|EndFunction|WeakExternal|\.bf or\.ef};

    my $file;
    foreach my $line ( @bulk ) {
        if ( $line =~ m{\ADump of file (.*?)\n\z} ) {
            $file = $1;
            if ( not exists( $objects->{ $file } ) ) {
                die "link reported unknown object file:\n    $line\n";
            }; # if
        } elsif ( $line =~ m{\A$num_re } ) {
            if ( not defined( $file ) ) {
                die "link reported symbol of unknown object file:\n    $line\n";
            }; # if
            if ( $line !~ m{\A$num_re $addr_re ($tag_re)\s+notype(?: \(\))?\s+($class_re)\s+\| (.*?)\n\z} ) {
                die "Cannot parse link output, line:\n    $line\n";
            }; # if
            my ( $tag, $class, $symbol ) = ( $1, $2, $3 );
            # link.exe /dump sometimes prints comments for symbols, e. g.:
            # ".?0_memcopyA ([Entry] ?0_memcopyA)", or "??_C@_01A@r?$AA@ (`string')".
            # Strip these comments.
            $symbol =~ s{ \(.*\)\z}{};
            if ( $class eq "External" ) {
                if ( $tag eq "UNDEF" ) { # Symbol not defined.
                    push( @{ $objects->{ $file }->{ undefined } }, $symbol );
                } else {                 # Symbol defined.
                    push( @{ $objects->{ $file }->{ defined } }, $symbol );
                }; # if
            }; # if
        } else {
            # Ignore all other lines.
        }; # if
    }; # foreach

    return undef;

}; # sub _load_symbols_link

# --------------------------------------------------------------------------------------------------
# Name:
#     symbols -- Construct set of symbols with specified tag in the specified set of objects.
# Synopsis:
#     my $symbols = defined_symbols( $objects, $tag );
# Arguments:
#     $objects (in) -- Set of objects.
#     $tag (in) -- A tag, "defined" or "undefined".
# Returns:
#     Set of symbols with the specified tag.
#
sub symbols($$) {

    my $objects = shift( @_ );
    my $tag     = shift( @_ );

    my $symbols = {};

    foreach my $object ( keys( %$objects ) ) {
        foreach my $symbol ( @{ $objects->{ $object }->{ $tag } } ) {
            push( @{ $symbols->{ $symbol } }, $object );
        }; # foreach $symbol
    }; # foreach $object

    return $symbols;

}; # sub symbols

sub defined_symbols($) {

    my $objects = shift( @_ );
    my $defined = symbols( $objects, "defined" );
    return $defined;

}; # sub defined_symbols

sub undefined_symbols($) {

    my $objects = shift( @_ );
    my $defined = symbols( $objects, "defined" );
    my $undefined = symbols( $objects, "undefined" );
    foreach my $symbol ( keys( %$defined ) ) {
        delete( $undefined->{ $symbol } );
    }; # foreach symbol
    return $undefined;

}; # sub undefined_symbols

# --------------------------------------------------------------------------------------------------
# Name:
#     _required_extra_objects -- Select a subset of extra objects required to resolve undefined
#         symbols in a set of objects. It is a helper sub for required_extra_objects().
# Synopsis:
#     my $required = _required_extra_objects( $objects, $extra, $symbols );
# Arguments:
#     $objects (in) -- A set of objects to be searched for undefined symbols.
#     $extra (in) -- A set of extra objects to be searched for defined symbols to resolve undefined
#         symbols in objects.
#     $symbols (in/out) -- Set of symbols defined in the set of external objects. At the first call
#         it should consist of all the symbols defined in all the extra objects. Symbols defined in
#         the selected subset of extra objects are removed from set of defined symbols, because
#         they are out of interest for subsequent calls.
# Returns:
#     A subset of extra objects required by the specified set of objects.
#
sub _required_extra_objects($$$$) {

    my $objects = shift( @_ );
    my $extra   = shift( @_ );
    my $symbols = shift( @_ );
    my $dump    = shift( @_ );

    my $required = {};

    if ( $dump > 0 ) {
        STDERR->print( "Required extra objects:\n" );
    }; # if
    foreach my $object ( keys( %$objects ) ) {
        foreach my $symbol ( @{ $objects->{ $object }->{ undefined } } ) {
            if ( exists( $symbols->{ $symbol } ) ) {
                # Add all objects where the symbol is defined to the required objects.
                foreach my $req_obj ( @{ $symbols->{ $symbol } } ) {
                    if ( $dump > 0 ) {
                        STDERR->print( "    $req_obj\n" );
                        if ( $dump > 1 ) {
                            STDERR->print( "        by $object\n" );
                            STDERR->print( "            due to $symbol\n" );
                        }; # if
                    }; # if
                    $required->{ $req_obj } = $extra->{ $req_obj };
                }; # foreach $req_obj
                # Delete the symbol from list of defined symbols.
                delete( $symbols->{ $symbol } );
            }; # if
        }; # foreach $symbol
    }; # foreach $object

    return $required;

}; # sub _required_extra_objects


# --------------------------------------------------------------------------------------------------
# Name:
#     required_extra_objects -- Select a subset of extra objects required to resolve undefined
#         symbols in a set of base objects and selected extra objects.
# Synopsis:
#     my $required = required_extra_objects( $base, $extra );
# Arguments:
#     $base (in/out) -- A set of base objects to be searched for undefined symbols. On enter, it is
#         expected that top-level hash has filled with object names only. On exit, it is completely
#         fulfilled with lists of symbols defined and/or referenced in each object file.
#     $extra (in/out) -- A set of extra objects to be searched for defined symbols required to
#         resolve undefined symbols in a set of base objects. Usage is similar to base objects.
# Returns:
#     A subset of extra object files.
#
sub required_extra_objects($$$) {

    my $base    = shift( @_ );
    my $extra   = shift( @_ );
    my $dump    = shift( @_ );

    # Load symbols for each object.
    load_symbols( $base );
    load_symbols( $extra );
    if ( $dump ) {
        dump_objects( "Base objects:", $base, $dump );
        dump_objects( "Extra objects:", $extra, $dump );
    }; # if

    # Collect symbols defined in extra objects.
    my $symbols = defined_symbols( $extra );

    my $required = {};
    # Select extra objects required by base objects.
    my $delta = _required_extra_objects( $base, $extra, $symbols, $dump );
    while ( %$delta ) {
        %$required = ( %$required, %$delta );
        # Probably, just selected objects require some more objects.
        $delta = _required_extra_objects( $delta, $extra, $symbols, $dump );
    }; # while

    if ( $dump ) {
        my $base_undefined = undefined_symbols( $base );
        my $req_undefined = undefined_symbols( $required );
        dump_symbols( "Symbols undefined in base objects:", $base_undefined, $dump );
        dump_symbols( "Symbols undefined in required objects:", $req_undefined, $dump );
    }; # if

    return $required;

}; # sub required_extra_objects


# --------------------------------------------------------------------------------------------------
# Name:
#     copy_objects -- Copy (and optionally edit) object files to specified directory.
# Synopsis:
#     copy_objects( $objects, $target, $prefix, @symbols );
# Arguments:
#     $objects (in) -- A set of object files.
#     $target (in) -- A name of target directory. Directory must exist.
#     $prefix (in) -- A prefix to add to all the symbols listed in @symbols. If prefix is undefined,
#         object files are just copied.
#     @symbols (in) -- List of symbol names to be renamed.
# Returns:
#     None.
#
sub copy_objects($$;$\@) {

    my $objects = shift( @_ );
    my $target  = shift( @_ );
    my $prefix  = shift( @_ );
    my $symbols = shift( @_ );
    my $tool;
    my @redefine;
    my @redefine_;
    my $syms_file = "__kmp_sym_pairs.log";

    if ( $target_arch eq "mic" ) {
        $tool = "x86_64-k1om-linux-objcopy"
    } else {
        $tool = "objcopy"
    }

    if ( not -e $target ) {
        die "\"$target\" directory does not exist\n";
    }; # if
    if ( not -d $target ) {
        die "\"$target\" is not a directory\n";
    }; # if

    if ( defined( $prefix ) and @$symbols ) {
        my %a = map ( ( "$_ $prefix$_" => 1 ), @$symbols );
        @redefine_ = keys( %a );
    }; # if
    foreach my $line ( @redefine_ ) {
        $line =~ s{$prefix(\W+)}{$1$prefix};
        push( @redefine, $line );
    }
    write_file( $syms_file, \@redefine );
    foreach my $src ( sort( keys( %$objects ) ) ) {
        my $dst = cat_file( $target, get_file( $src ) );
        if ( @redefine ) {
            execute( [ $tool, "--redefine-syms", $syms_file, $src, $dst ] );
        } else {
            copy_file( $src, $dst, -overwrite => 1 );
        }; # if
    }; # foreach $object

}; # sub copy_objects


# --------------------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------------------

my $base  = {};
my $extra = {};
my $switcher = $base;
my $dump = 0;
my $print_base;
my $print_extra;
my $copy_base;
my $copy_extra;
my $prefix;

# Parse command line.

Getopt::Long::Configure( "permute" );
get_options(
    Platform::target_options(),
    "base"         => sub { $switcher = $base;  },
    "extra"        => sub { $switcher = $extra; },
    "print-base"   => \$print_base,
    "print-extra"  => \$print_extra,
    "print-all"    => sub { $print_base = 1; $print_extra = 1; },
    "copy-base=s"  => \$copy_base,
    "copy-extra=s" => \$copy_extra,
    "copy-all=s"   => sub { $copy_base = $_[ 1 ]; $copy_extra = $_[ 1 ]; },
    "dump"         => sub { ++ $dump; },
    "prefix=s"     => \$prefix,
    "<>"    =>
        sub {
            my $arg = $_[ 0 ];
            my @args;
            if ( $^O eq "MSWin32" ) {
                # Windows* OS does not expand wildcards. Do it...
                @args = bsd_glob( $arg );
            } else {
                @args = ( $arg );
            }; # if
            foreach my $object ( @args ) {
                if ( exists( $base->{ $object } ) or exists( $extra->{ $object } ) ) {
                    die "Object \"$object\" has already been specified.\n";
                }; # if
                $switcher->{ $object } = { defined => [], undefined => [] };
            }; # foreach
        },
);
if ( not %$base ) {
    cmdline_error( "No base objects specified" );
}; # if

if ( $target_os eq "win" ) {
    *load_symbols = \&_load_symbols_link;
} elsif ( $target_os eq "lin" ) {
    *load_symbols = \&_load_symbols_nm;
} elsif ( $target_os eq "mac" ) {
    *load_symbols = \&_load_symbols_nm;
} else {
    runtime_error( "OS \"$target_os\" not supported" );
}; # if

# Do the work.

my $required = required_extra_objects( $base, $extra, $dump );
if ( $print_base ) {
    print( map( "$_\n", sort( keys( %$base ) ) ) );
}; # if
if ( $print_extra ) {
    print( map( "$_\n", sort( keys( %$required ) ) ) );
}; # if
my @symbols;
if ( defined( $prefix ) ) {
    foreach my $object ( sort( keys( %$required ) ) ) {
        push( @symbols, @{ $required->{ $object }->{ defined } } );
    }; # foreach $objects
}; # if
if ( $copy_base ) {
    copy_objects( $base, $copy_base, $prefix, @symbols );
}; # if
if ( $copy_extra ) {
    copy_objects( $required, $copy_extra, $prefix, @symbols );
}; # if

exit( 0 );

__END__

=pod

=head1 NAME

B<required-objects.pl> -- Select a required extra object files.

=head1 SYNOPSIS

B<required-objects.pl> I<option>... [--base] I<file>... --extra I<file>...

=head1 DESCRIPTION

B<required-objects.pl> works with two sets of object files -- a set of I<base> objects
and a set of I<extra> objects, and selects those extra objects which are required for resolving
undefined symbols in base objects I<and> selected extra objects.

Selected object files may be copied to specified location or their names may be printed to stdout,
a name per line. Additionally, symbols defined in selected extra objects may be renamed.

Depending on OS, different external tools may be used. For example, B<required-objects.pl> uses
F<link.exe> on "win" and F<nm> on "lin" and "mac" OSes. Normally OS is autodetected, but
detection can be overrided with B<--os> option. It may be helpful in cross-build environments.

=head1 OPTIONS

=over

=item B<--base>

The list of base objects follows this option.

=item B<--extra>

List of extra objects follows this option.

=item B<--print-all>

Print list of base objects and list of required extra objects.

=item B<--print-base>

Print list of base objects.

=item B<--print-extra>

Print list of selected extra objects.

=item B<--copy-all=>I<dir>

Copy all base and selected extra objects to specified directory. The directory must exist. Existing
files are overwritten.

=item B<--copy-base=>I<dir>

Copy all base objects to specified directory.

=item B<--copy-extra=>I<dir>

Copy selected extra objects to specified directory.

=item B<--prefix=>I<str>

If prefix is specified, copied object files are edited -- symbols defined in selected extra
object files are renamed (in all the copied object files) by adding this prefix.

F<objcopy> program should be available for performing this operation.

=item B<--os=>I<str>

Specify OS name. By default OS is autodetected.

Depending on OS, B<required-objects.pl> uses different external tools.

=item B<--help>

Print short help message and exit.

=item B<--doc>

=item B<--manual>

Print full documentation and exit.

=item B<--version>

Print version and exit.

=back

=head1 ARGUMENTS

=over

=item I<file>

A name of object file.

=back

=head1 EXAMPLES

    $ required-objects.pl --base obj/*.o --extra ../lib/obj/*.o --print-extra > required.lst
    $ ar cr libx.a obj/*.o $(cat required.lst)

    $ required-objects.pl --base internal/*.o --extra external/*.o --prefix=__xyz_ --copy-all=obj
    $ ar cr xyz.a obj/*.o

=cut

# end of file #

