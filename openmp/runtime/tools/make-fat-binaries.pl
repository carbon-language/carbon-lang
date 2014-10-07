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

use IO::Dir;

use FindBin;
use lib "$FindBin::Bin/lib";


use tools;

our $VERSION = "0.003";

#
# Subroutines.
#

sub check_dir($$) {

    # Make sure a directory is a readable directory.

    my ( $dir, $type ) = @_;

    -e $dir or runtime_error( "Directory \"$dir\" does not exist" );
    -d $dir or runtime_error( "\"$dir\" is not a directory" );
    -r $dir or runtime_error( "Directory \"$dir\" is not readable" );

}; # sub check_dir

sub read_dir($) {

    # Return list of files (not subdirectories) of specified directory.

    my ( $dir ) = @_;
    my $handle;
    my $entry;
    my @files;

    $handle = IO::Dir->new( $dir ) or runtime_error( "Cannot open \"$dir\" directory: $!" );
    while ( $entry = $handle->read() ) {
        my $path = "$dir/$entry";
        if ( $entry !~ m{\A\.} and -f $path ) {
            push( @files, $entry );
        }; # if
    }; # while
    $handle->close();

    @files = sort( @files );
    return @files;

}; # sub read_dir

# --------------------------------------------------------------------------------------------------
# Main program.
# --------------------------------------------------------------------------------------------------

#
# Parse command line.
#
my @dirs;    # List of input directories.
my @files;   # List of files.
my $output;  # Output directory.

get_options(
    "output=s" => \$output
);

@ARGV == 0 and cmdline_error( "No input directories specified" );

#
# Check input and output directories.
#

# Make shure there is no duplicated directories.
my %dirs;
$dirs{ $output } = "";
foreach my $dir ( @ARGV ) {
    if ( exists( $dirs{ $dir } ) ) {
        cmdline_error( "Directory \"$dir\" has already been specified" );
    }; # if
    $dirs{ $dir } = "";
    push( @dirs, $dir );
}; # foreach $dir
undef( %dirs );

# Make sure all dirs are exist, dirs, and readable.
check_dir( $output, "output" );
foreach my $dir ( @dirs ) {
    check_dir( $dir,  "input" );
}; # foreach $dir

# All input dirs should contain exactly the same list of files.
my @errors;
@files = read_dir( $dirs[ 0 ] );
foreach my $dir ( @dirs ) {
    my %files = map( ( $_ => 0 ), @files );
    foreach my $file ( read_dir( $dir ) ) {
        if ( not exists( $files{ $file } ) ) {
            push( @errors, "Extra file: `" . cat_file( $dir, $file ) . "'." );
        }; # if
        $files{ $file } = 1;
    }; # foreach $file
    foreach my $file ( keys( %files ) ) {
        if ( $files{ $file } == 0 ) {
            push( @errors, "Missed file: `" . cat_file( $dir, $file ) . "'." );
        }; # if
    }; # foreach $file
}; # foreach $dir
if ( @errors ) {
    runtime_error( @errors );
}; # if

#
# Make fat binaries.
#

foreach my $file ( sort( @files ) ) {
    info( "Making \"$file\"..." );
    my $output_file = cat_file( $output, $file );
    del_file( $output_file );
    execute(
        [
            "lipo",
            "-create",
            "-output", $output_file,
            map( cat_file( $_, $file ), @dirs )
        ]
    );
}; # foreach $entry

exit( 0 );

__END__

=pod

=head1 NAME

B<make-fat-binaries.pl> -- Make set of fat (universal) binaries.

=head1 SYNOPSIS

B<make-fat-binaries.pl> I<OPTION>... I<INPUT_DIR>...

=head1 OPTIONS

=over

=item B<--output=>I<DIR>

Name of output directory to place fat binaries to. Directory must exist and be writable.

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

=item I<INPUT_DIR>

Name of input directory to get thin files from. Directory must exist and be readable. At least one
directory required.

=back

=head1 DESCRIPTION

The script creates set of Mac-O fat (universal, multi-architecture) binaries from set of thin
(single-architecture) files.

The scripts reads files from input directory (or directoriers). It is assumed that one input
directory keeps files for one architecture (e. g. i386), another directory contains files for
another architecture (e. g. x86_64), etc. All input directories must contain the same set of files.
The script issues an error if sets of files in input directories differ.

If the script finishes successfuly, output directory will contain the set universal binaries
built from files with the same name in input directories.

=head1 EXAMPLES

Get thin binaries from C<mac_32.thin/> and C<mac_32e.thin/> directories, and put fat binaries to
C<mac.fat/> directory:

    $ make-fat-binaries.pl --output=mac.fat mac_32.thin mac_32e.thin


=cut

# end of file #
