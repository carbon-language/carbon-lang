#!/usr/bin/env perl

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
use File::Temp;
use Cwd;

use FindBin;
use lib "$FindBin::Bin/lib";

use tools;
use Uname;
use Platform ":vars";

our $VERSION = "0.005";

# --------------------------------------------------------------------------------------------------
# Subroutines.
# --------------------------------------------------------------------------------------------------

sub windows {
    my ( $arch, $output, @args ) = @_;
    my %files;
    # TODO: Check the archives are of specified architecture.
    foreach my $arg ( @args ) {
        foreach my $archive ( bsd_glob( $arg ) ) {
            info( "Processing \"$archive\"..." );
            my $bulk;
            execute( [ "lib.exe", "/nologo", "/list", $archive ], -stdout => \$bulk );
            my @members = split( "\n", $bulk );
            foreach my $member ( @members ) {
                my $file = get_file( $member );
                my $path = cat_file( $output, $file );
                if ( exists( $files{ $file } ) ) {
                    runtime_error(
                        "Extraction \"$file\" member from \"$archive\" archive failed:",
                        "\"$file\" member has already been extracted from \"$files{ $file }\" archive"
                    );
                }; # if
                $files{ $file } = $archive;
                info( "    Writing \"$path\"..." );
                execute( [ "lib.exe", "/nologo", "/extract:" . $member, "/out:" . $path, $archive ] );
            }; # foreach $member
        }; # foreach $archive
    }; # foreach $arg
}; # sub windows

sub linux {
    my ( $arch, $output, @archives ) = @_;
    # TODO: Check the archives are of specified architecture.
    my $cwd = Cwd::cwd();
    change_dir( $output );
    foreach my $archive ( @archives ) {
        info( "Processing \"$archive\"..." );
        my $path = abs_path( $archive, $cwd );
        execute( [ "ar", "xo", $path ] );
    }; # foreach $archive
    change_dir( $cwd );
}; # sub linux

my %mac_arch = (
    "32"  => "i386",
    "32e" => "x86_64"
);

sub darwin {
    my ( $arch, $output, @archives ) = @_;
    my $cwd = getcwd();
    change_dir( $output );
    if ( defined( $arch ) ) {
        if ( not defined( $mac_arch{ $arch } ) ) {
            runtime_error( "Architecture \"$arch\" is not a valid one for OS X*" );
        }; # if
        $arch = $mac_arch{ $arch };
    }; # if
    foreach my $archive ( @archives ) {
        info( "Processing \"$archive\"..." );
        my $path = abs_path( $archive, $cwd );
        my $temp;
        # Whether archive is a fat or thin?
        my $bulk;
        execute( [ "file", $path ], -stdout => \$bulk );
        if ( $bulk =~ m{Mach-O universal binary} ) {
            # Archive is fat, extracy thin archive first.
            if ( not defined( $arch ) ) {
                runtime_error(
                    "\"$archive\" archive is universal binary, " .
                        "please specify architecture to work with"
                );
            }; # if
            ( undef, $temp ) = File::Temp::tempfile();
            execute( [ "libtool", "-static", "-arch_only", $arch, "-o", $temp, $path ] );
            $path = $temp;
        }; # if
        execute( [ "ar", "xo", $path ] );     # Extract members.
        if ( defined( $temp ) ) {             # Delete temp file, if any.
            del_file( $temp );
        }; # if
    }; # foreach $archive
    change_dir( $cwd );
}; # sub darwin


# --------------------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------------------

# Parse command line.

my $output = ".";
my @args;

get_options(
    Platform::target_options(),
    "o|output-directory=s" => \$output,
);
@args = @ARGV;

if ( not -e $output ) {
    runtime_error( "Output directory \"$output\" does not exist" );
}; # if
if ( not -d $output ) {
    runtime_error( "\"$output\" is not a directory" );
}; # if
if ( not -w $output ) {
    runtime_error( "Output directory \"$output\" is not writable" );
}; # if

if ( $target_os eq "win" ) {
    *process = \&windows;
} elsif ( $target_os eq "lin" or $target_os eq "lrb" ) {
    *process = \&linux;
} elsif ( $target_os eq "mac" ) {
    *process = \&darwin;
} else {
    runtime_error( "OS \"$target_os\" not supported" );
}; # if


# Do the work.
process( $target_arch, $output, @args );
exit( 0 );

__END__

=pod

=head1 NAME

B<extract-objects.pl> -- Extract all object files from static library.

=head1 SYNOPSIS

B<extract-objects.pl> I<option>... I<archive>...

=head1 OPTIONS

=over

=item B<--architecture=>I<arch>

Specify architecture to work with. The option is mandatory on OS X* in case of universal archive.
In other cases the option should not be used. I<arch> may be one of C<32> or C<32e>.

=item B<--os=>I<str>

Specify OS name. By default OS is autodetected.

Depending on OS, B<extract-objects.pl> uses different external tools for handling static
libraries: F<ar> (in case of "lin" and "mac") or F<lib.exe> (in case of "win").

=item B<--output-directory=>I<dir>

Specify directory to write extracted members to. Current directory is used by default.

=item B<--help>

Print short help message and exit.

=item B<--doc>

=item B<--manual>

Print full documentation and exit.

=item B<--quiet>

Do not print information messages.

=item B<--version>

Print version and exit.

=back

=head1 ARGUMENTS

=over

=item I<archive>

A name of archive file (static library). Multiple archives may be specified.

=back

=head1 DESCRIPTION

The script extracts all the members (object files) from archive (static library) to specified
directory. Commands to perform this action differ on different OSes. On Linux* OS, simple command

    ar xo libfile.a

is enough (in case of extracting files to current directory).

On OS X*, it is a bit compilicated with universal ("fat") binaries -- C<ar> cannot
operate on fat archives, so "thin" archive should be extracted from the universal binary first.

On Windows* OS, library manager (C<lib.exe>) can extract only one object file, so operation should be
repeated for every object file in the library.

B<extract-objects.pl> detects OS automatically. But detection can be overrided with B<--os> option.
It may be helpful in cross-build environments.

B<extract-objects.pl> effectively encapsulates all these details and provides uniform way for
extracting object files from static libraries, which helps to keep makefiles simple and clean.

=head1 EXAMPLES

Extract object files from library F<libirc.lib>, and put them into F<obj/> directory:

    $ extract-objects.pl --output=obj libirc.lib

Extract object files from library F<libirc.a>. Use Linux* OS tools (F<ar>), even if run on another OS:

    $ extract-objects.pl --os=lin libirc.a

Extract object files from library F<libirc.a>, if it is a OS X* universal binary, use i386
architecture. Be quiet:

    $ extract-objects.pl --quiet --arch=i386 libirc.a

=cut

# end of file #

