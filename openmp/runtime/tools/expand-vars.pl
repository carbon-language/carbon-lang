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

use FindBin;
use lib "$FindBin::Bin/lib";

use tools;

our $VERSION = "0.005";

my $name_rexp    = qr{[A-Za-z_]+[A-Za-z0-9_]*};
my $keyword_rexp = qr{if|else|end|omp};

sub error($$$) {
    my ( $input, $msg, $bulk ) = @_;
    my $pos = pos( $$bulk );
    $$bulk =~ m{^(.*?)\G(.*?)$}m or die "Internal error";
    my ( $pre, $post ) = ( $1, $2 );
    my $n = scalar( @{ [ substr( $$bulk, 0, $pos ) =~ m{\n}g ] } ) + 1;
    runtime_error( "\"$input\" line $n: $msg:", ">>> " . $pre . "--[HERE]-->" . $post );
}; # sub error

sub evaluate($$$\$) {
    my ( $expr, $strict, $input, $bulk ) = @_;
    my $value;
    { # Signal handler will be restored on exit from this block.
        # In case of "use strict; use warnings" eval issues warnings to stderr. This direct
        # output may confuse user, so we need to catch it and prepend with our info.
        local $SIG{ __WARN__ } = sub { die @_; };
        $value =
            eval(
                "package __EXPAND_VARS__;\n" .
                ( $strict ? "use strict; use warnings;\n" : "no strict; no warnings;\n" ) .
                $expr
            );
    };
    if ( $@ ) {
        # Drop location information -- increasing eval number and constant "line 3"
        # is useless for the user.
        $@ =~ s{ at \(eval \d+\) line \d+}{}g;
        $@ =~ s{\s*\z}{};
        error( $input, "Cannot evaluate expression \"\${{$expr}}\": $@", $bulk );
    }; # if
    if ( $strict and not defined( $value ) ) {
        error( $input, "Substitution value is undefined", $bulk );
    }; # if
    return $value;
}; # sub evaluate

#
# Parse command line.
#

my ( @defines, $input, $output, $strict );
get_options(
    "D|define=s" => \@defines,
    "strict!"    => \$strict,
);
if ( @ARGV < 2 ) {
    cmdline_error( "Not enough argument" );
}; # if
if ( @ARGV > 2 ) {
    cmdline_error( "Too many argument(s)" );
}; # if
( $input, $output ) = @ARGV;

foreach my $define ( @defines ) {
    my ( $equal, $name, $value );
    $equal = index( $define, "=" );
    if ( $equal < 0 ) {
        $name = $define;
        $value = "";
    } else {
        $name = substr( $define, 0, $equal );
        $value = substr( $define, $equal + 1 );
    }; # if
    if ( $name eq "" ) {
        cmdline_error( "Illegal definition: \"$define\": variable name should not be empty." );
    }; # if
    if ( $name !~ m{\A$name_rexp\z} ) {
        cmdline_error(
            "Illegal definition: \"$define\": " .
                "variable name should consist of alphanumeric characters."
        );
    }; # if
    eval( "\$__EXPAND_VARS__::$name = \$value;" );
    if ( $@ ) {
        die( "Internal error: $@" );
    }; # if
}; # foreach $define

#
# Do the work.
#

my $bulk;

# Read input file.
$bulk = read_file( $input );

# Do the replacements.
$bulk =~
    s{(?:\$($keyword_rexp)|\$($name_rexp)|\${{(.*?)}})}
    {
        my $value;
        if ( defined( $1 ) ) {
            # Keyword. Leave it as is.
            $value = "\$$1";
        } elsif ( defined( $2 ) ) {
            # Variable to expand.
            my $name = $2;
            $value = eval( "\$__EXPAND_VARS__::$name" );
            if ( $@ ) {
                die( "Internal error" );
            }; # if
            if ( $strict and not defined( $value ) ) {
                error( $input, "Variable \"\$$name\" not defined", \$bulk );
            }; # if
        } else {
            # Perl code to evaluate.
            my $expr = $3;
            $value = evaluate( $expr, $strict, $input, $bulk );
        }; # if
        $value;
    }ges;

# Process conditionals.
# Dirty patch! Nested conditionals not supported!
# TODO: Implement nested constructs.
$bulk =~
    s{^\$if +([^\n]*) *\n(.*\n)\$else *\n(.*\n)\$end *\n}
    {
        my ( $expr, $then_part, $else_part ) = ( $1, $2, $3 );
        my $value = evaluate( $expr, $strict, $input, $bulk );
        if ( $value ) {
            $value = $then_part;
        } else {
            $value = $else_part;
        }; # if
    }gesm;

# Write output.
write_file( $output, \$bulk );

exit( 0 );

__END__

=pod

=head1 NAME

B<expand-vars.pl> -- Simple text preprocessor.

=head1 SYNOPSIS

B<expand-vars.pl> I<OPTION>... I<input> I<output>

=head1 OPTIONS

=over

=item B<-D> I<name>[B<=>I<value>]

=item B<--define=>I<name>[B<=>I<value>]

Define variable.

=item B<--strict>

In strict mode, the script issues error on using undefined variables and executes Perl code
with C<use strict; use warnings;> pragmas.

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

=head1 ARGUMENTS

=over

=item I<input>

Input file name.

=item I<output>

Output file name.

=back

=head1 DESCRIPTION

This script reads input file, makes substitutes and writes output file.

There are two form of substitutes:

=over

=item Variables

Variables are referenced in input file in form:

    $name

Name of variable should consist of alphanumeric characters (Latin letters, digits, and underscores).
Variables are defined in command line with C<-D> or C<--define> options.

=item Perl Code

Perl code is specified in input file in form:

    ${{ ...code... }}

The code is evaluated, and is replaced with its result. Note: in strict mode, you should declare
variable before use. See examples.

=back

=head1 EXAMPLES

Replace occurrences of C<$year>, C<$month>, and C<$day> in C<input.txt> file with C<2007>, C<09>, C<01>
respectively and write result to C<output.txt> file:

    $ cat input.var
    Today is $year-$month-$day.
    $ expand-vars.pl -D year=2007 -D month=09 -D day=01 input.var output.txt && cat output.txt
    Today is 2007-09-01.

Using Perl code:

    $ cat input.var
    ${{ localtime(); }}
    $ expand-vars.pl -D year=2007 -D month=09 -D day=01 input.var output.txt && cat output.txt
    Now Tue May  5 20:54:13 2009

Using strict mode for catching bugs:

    $ cat input.var
    ${{ "year : " . substr( $date, 0, 4 ); }}
    $ expand-vars.pl input.var output.txt && cat output.txt
    year :

Oops, why it does not print year? Let us use strict mode:

    $ expand-vars.pl --strict input.var output.txt && cat output.txt
    expand-vars.pl: (x) "test.var": Cannot evaluate expression "${{ "year : " . substr( $date, 0, 4 ); }}": Global symbol "$date" requires explicit package name

Ok, variable is not defined. Let us define it:

    $ expand-vars.pl --strict -D date=20090501 input.var output.txt && cat output.txt
    expand-vars.pl: (x) "test.var": Cannot evaluate expression "${{ "year : " . substr( $date, 0, 4 ); }}": Variable "$date" is not imported

What is wrong? Variable should be declared:

    $ cat input.var
    ${{ our $date; "year : " . substr( $date, 0, 4 ); }}
    $ expand-vars.pl --strict -D date=20090501 input.var output.txt && cat output.txt
    year : 2009

=cut

# end of file #
