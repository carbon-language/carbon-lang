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

use FindBin;
use lib "$FindBin::Bin/lib";

use tools;

our $VERSION = "0.02";

sub wipe($$$) {

    my ( $input, $output, $wipe ) = @_;
    my $bulk = read_file( $input, -binary => 1 );
    $bulk =~ s{($wipe)}{ " " x length( $1 ) }ge;
    write_file( $output, \$bulk, -binary => 1 );
    return undef;

}; # sub wipe

my @wipe;
my $target = ".";
get_options(
    "wipe-literal=s"     =>
        sub { my $arg = $_[ 1 ]; push( @wipe, qr{@{ [ quotemeta( $arg ) ] }} ); },
    "wipe-regexp=s"      =>
        sub { my $arg = $_[ 1 ]; push( @wipe, qr{$arg} ); },
    "target-directory=s" => \$target,
);

# Convert strings to regular expression.
my $wipe = qr{@{ [ join( "|", @wipe ) ] }};

my %jobs;

# Collect files to process.
# jobs: output -> input.
foreach my $arg ( @ARGV ) {
    my @inputs = ( $^O eq "MSWin32" ? bsd_glob( $arg ) : ( $arg ) );
    foreach my $input ( @inputs ) {
        my $file   = get_file( $input );
        my $output = cat_file( $target, $file );
        if ( exists( $jobs{ $output } ) ) {
            runtime_error(
                "\"$jobs{ $output }\" and \"$input\" input files tend to be written " .
                    "to the same output file \"$output\""
            );
        }; # if
        $jobs{ $output } = $input;
    }; # foreach
}; # foreach $file

# Process files.
%jobs = reverse( %jobs ); # jobs: input -> output.
foreach my $input ( sort( keys( %jobs ) ) ) {
    my $output = $jobs{ $input };
    info( "\"$input\" -> \"$output\"" );
    wipe( $input, $output, $wipe );
}; # foreach $input

exit( 0 );

__END__

#
# Embedded documentation.
#

=pod

=head1 NAME

B<wipe-string.pl> -- Wipe string in text or binary files.

=head1 SYNOPSIS

B<wipe-string.pl> I<OPTION>... I<FILE>...

=head1 OPTIONS

=over

=item B<--doc>

=item B<--manual>

Print full help message and exit.

=item B<--help>

Print short help message and exit.

=item B<--target-directory=>I<dir>

Directory to put result files to. By default result files are written in the current working
directory.

=item B<--usage>

Print very short usage message and exit.

=item B<--version>

Print version and exit.

=item B<--wipe-literal=>I<str>

Specify literal string to wipe. Multiple literals are allowed.

=item B<--wipe-regexp=>I<str>

Specify Perl regular expression to wipe. Multiple regular expressions may be specified.

Be careful. Protect special characters from beign interpreted by shell.

=back

=head1 ARGUMENTS

=over

=item I<file>

File name to wipe string in.

=back

=head1 DESCRIPTION

The script wipes strings in files. String may be specified literally or by Perl regular expression.
Strings are wiped by replacing characters with spaces, so size of file remains the same. The script
may be applied to both text and binary files.

Result files are written by default to current directory, or to directory specified by
B<--target-directory> option, if any. If multiple input files tend to be written to the same output
file (e. g. identically named input files located in different directories), the script generates an
error.

The script reads entire file, process it, and the writes to disk. Therefore it is (almost) safe to
update files in-place (see examples).

=head1 EXAMPLES

Wipe "Copyright" word in all the files with "txt" suffix in current directory, overwrite original
files (update them in-place):

    wipe-string.pl --wipe-literal="Copyright" *.txt

Wipe "Copyright" and "Copyleft" words in all the files with "txt" suffix in current directory,
write result files to ../wiped directory:

    wipe-string.pl --wipe-literal=Copyright --wipe-literal=Copyleft --target-dir=../wiped *.txt

Wipe "Copyright" and "Copyleft" words in files from "doc" directory, write result files to current
directory;

    wipe-string.pl --wipe-regexp="Copyright|Copyleft" doc/*

Wipe "defaultlib" directive in all the library files:

    wipe-string.pl --wipe-regexp="-defaultlib:[A-Za-z0-9_.]+" *.lib

(Be careful: the script does not analyze structure of library and object files, it just wipes
U<strings>, so it wipes all the occurrences of strings matching to specified regular expression.)

=cut

# end of file #
