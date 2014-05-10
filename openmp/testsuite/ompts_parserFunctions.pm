#!/usr/bin/perl -w

# functions.pm
# This package contains a set of subroutines to modify the templates for the openMP Testuite.


################################################################################
# subroutines to extract, modify or delete tags from the template
################################################################################

# LIST get_tag_values( $tagname, $string )
# subrutine to get the text encloded by a tag.
# Returns a list containing the inner texts of the found tags
sub get_tag_values
{
	my ( $tagname, $string );
	( $tagname, $string ) = @_;
	my (@tmp,@tmp2);
   	@tmp = split(/\<$tagname\>/,$string); 
	foreach $_(@tmp){
		push(@tmp2,split(/\<\/$tagname\>/));
	}
	my(@result,$i);
	$i=1; # couter to get only every second item
	foreach $_(@tmp2){
		if($i%2 eq 0){
			push(@result,$_);
		}
		$i++;
	}
	return @result;
}

# LIST replace_tags( $tagname, $replacestring, @list )
# subrutine to replace tags by a replacestring. 
# Returns a list of the srings after conversion.
sub replace_tags
{
	my ($tagname, $replacestring, @stringlist, @result);
	($tagname, $replacestring, @stringlist) = @_;
	foreach $_(@stringlist) {
		s#\<$tagname\>(.*?)\<\/$tagname\>#$replacestring#gs;
		push(@result,$_);
	}
	return @result;
}

# LIST enlarge_tags( $tagname, $before, $after, @list )
# subrutine to replace tags by the tags added by a string before and after. 
# Returns a list of the srings after conversion.
sub enlarge_tags
{
	my ($tagname, $before, $after, @stringlist,@result);
	($tagname, $before, $after, @stringlist) = @_;
	foreach $_(@stringlist) {
		s#\<$tagname\>(.*?)\<\/$tagname\>#$before$1$after#gs;
		push(@result,$_);
	}
	return @result;
}

# LIST delete_tags( $tagname, @list )
# subrutine to delete tags in a string. 
# Returns a list of the cleared strings
sub delete_tags
{
	my($tagname,@stringlist);
	($tagname, @stringlist) = @_;
	my(@result);
	foreach $_(@stringlist) {
		s#\<$tagname\>(.*?)\<\/$tagname\>##gs;
		push(@result,$_);
	}
	return @result;
}



################################################################################
# subroutines for generating "orpahned" tests 					
################################################################################

# SCALAR create_orph_cfunctions( $prefix, $code )
# returns a string containing the definitions of the functions for the 
# orphan regions.
sub create_orph_cfunctions
{
	my ($code,@defs);
	($code) = @_;
	@defs = get_tag_values('ompts:orphan',$code);
	($functionname) = get_tag_values('ompts:testcode:functionname',$code);
	my ( @result,$functionsrc, $i);
	$functionsrc =  "\n/* Automatically generated definitions of the orphan functions */\n";

	$i = 1;
	foreach (@defs) {
		$functionsrc .= "\nvoid orph$i\_$functionname (FILE * logFile) {";
		$functionsrc .= $_;
		$functionsrc .= "\n}\n";
		$i++;
	}
	$functionsrc .= "/* End of automatically generated definitions */\n";
	return $functionsrc;
}

# SCALAR create_orph_fortranfunctions( $prefix, $code )
# returns a string containing the definitions of the functions for the 
# orphan regions.
sub create_orph_fortranfunctions
{
	my ($prefix,$code,@defs,$orphan_parms);
	($prefix,$code,$orphan_parms) = @_;
	@defs = get_tag_values('ompts:orphan',$code);

    #to remove space and put a single space
    if($orphan_parms ne "")
    {
      $orphan_parms =~ s/[ \t]+//sg;
      $orphan_parms =~ s/[ \t]+\n/\n/sg;
    }
    
	($orphanvarsdefs) = get_tag_values('ompts:orphan:vars',$code);
	foreach (@varsdef) {
		if (not /[^ \n$]*/){ $orphanvarsdefs = join("\n",$orphanvarsdef,$_);}
	}
	($functionname) = get_tag_values('ompts:testcode:functionname',$code);
	my ( @result,$functionsrc, $i);
	$functionsrc =  "\n! Definitions of the orphan functions\n";
	$i = 1;
	foreach $_(@defs)
	{
		$functionsrc .= "\n      SUBROUTINE orph$i\_$prefix\_$functionname\($orphan_parms\)\n      ";
        $functionsrc .= "INCLUDE \"omp_testsuite.f\"\n";
		$functionsrc .= $orphanvarsdefs."\n";
		$functionsrc .= $_;
		$functionsrc .= "\n";
		$functionsrc .= "      END SUBROUTINE\n! End of definition\n\n";
		$i++;
	}
	return $functionsrc;
}

# LIST orphan_regions2cfunctions( $prefix, @code )
# replaces orphan regions by functioncalls in C/C++.
sub orphan_regions2cfunctions
{
	my ($code, $i, $functionname);
	($code) = @_;
	$i = 1;
	($functionname) = get_tag_values('ompts:testcode:functionname',$code);
        while( /\<ompts\:orphan\>(.*)\<\/ompts\:orphan\>/s) {
            s#\<ompts\:orphan\>(.*?)\<\/ompts\:orphan\>#orph$i\_$functionname (logFile);#s;
            $i++;
        }
	return $code;
}

# LIST orphan_regions2fortranfunctions( $prefix, @code )
# replaces orphan regions by functioncalls in fortran
sub orphan_regions2fortranfunctions
{
	my ( $prefix, @code, $my_parms, $i, $functionname);
	($prefix, ($code), $my_parms) = @_;
	$i = 1;
	($functionname) = get_tag_values('ompts:testcode:functionname',$code);
	foreach $_(($code))
	{
		while( /\<ompts\:orphan\>(.*)\<\/ompts\:orphan\>/s)
		{
			s#\<ompts\:orphan\>(.*?)\<\/ompts\:orphan\>#      CALL orph$i\_$prefix\_$functionname\($my_parms\);#s;
			$i++;
		}
	}
	return ($code);
}

# SCALAR orph_functions_declarations( $prefix, $code )
# returns a sring including the declaration of the functions used 
# in the orphan regions. The function names are generated using 
# the $prefix as prefix for the functionname.
sub orph_functions_declarations
{
	my ($prefix, $code);
	($prefix, $code) = @_;
	my ( @defs, $result );
	
	# creating declarations for later used functions
	$result .= "\n\n/* Declaration of the functions containing the code for the orphan regions */\n#include <stdio.h>\n";
	@defs = get_tag_values('ompts:orphan',$code);
	my ($functionname,$i);
	($functionname) = get_tag_values('ompts:testcode:functionname',$code);
	$i = 1;
	foreach $_(@defs) {
		$result .= "\nvoid orph$i\_$prefix\_$functionname ( FILE * logFile );";
		$i++;
	}
	$result .= "\n\n/* End of declaration */\n\n";
	return $result;
}

# SCALAR make_global_vars_definition( $code )
# returns a sring including the declaration for the vars needed to
# be declared global for the orphan region.
sub make_global_vars_def
{
	my ( $code );
	($code) = @_;
	my ( @defs, $result, @tmp, @tmp2 ,$predefinitions);
	
	# creating global declarations for the variables.
	$result  = "\n\n/* Declaration of the variables used in the orphan region. */\n";
	
	# get all tags containing the variable definitions
	@defs = get_tag_values('ompts:orphan:vars',$code);
	foreach $_(@defs)
	{
		# cutting the different declarations in the same tag by the ';' as cuttmark
		@tmp = split(/;/,$_);
		foreach $_(@tmp)
		{
			# replacing newlines and double spaces
			s/\n//gs;
			s/  //gs;
			# put the new declaration at the end of $result
			if($_ ne ""){ $result .= "\n $_;"; }
		}
	}
	$result .= "\n\n/* End of declaration. */\n\n";
	return $result;
}

# SCALAR extern_vars_definition( $code )
# returns a sring including the declaration for the vars needed to
# be declared extern for the orphan region.
sub extern_vars_def
{
	my ( $code );
	($code) = @_;
	my ( @defs, $result, @tmp, @tmp2 ,$predefinitions);
	
	# creating declarations for the extern variables.
	$result  = "\n\n/* Declaration of the extern variables used in the orphan region. */\n";
	# $result .= "\n#include <stdio.h>\n#include <omp.h>\n";
	$result .= "\nextern FILE * logFile;";
	
	# get all tags containing the variable definitions
	@defs = get_tag_values('ompts:orphan:vars',$code);
	foreach $_(@defs)
	{
		# cutting the different declarations in the same tag by the ';' as cuttmark
		@tmp = split(/;/,$_);
		foreach $_(@tmp)
		{
			# replacing newlines and double spaces
			s/\n//gs;
			s/  //gs;
			# cutting off definitions
			@tmp2 = split("=",$_);
			# put the new declaration at the end of $result
			$result .= "\nextern $tmp2[0];";
		}
	}
	$result .= "\n\n/* End of declaration. */\n\n";
	return $result;
}

sub leave_single_space
{
  my($str);
  ($str)=@_;
  if($str ne "")
  {
    $str =~ s/^[ \t]+/ /;
    $str =~ s/[ \t]+\n$/\n/;
    $str =~ s/[ \t]+//g;
  }
  return $str;
}

return 1;
