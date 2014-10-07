/*
 * $Revision: 42951 $
 * $Date: 2014-01-21 14:41:41 -0600 (Tue, 21 Jan 2014) $
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include <stdlib.h>
#include <iostream>
#include <strstream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <cstring>

using namespace std;

typedef std::string                      string_t;
typedef std::vector< string_t >          strings_t;
typedef std::map< string_t, string_t >   str_hash_t;
typedef std::pair< string_t, string_t >  str_pair_t;
#ifdef _WIN32
    typedef long long  int64_t;
#endif

string_t
shift( strings_t & strs ) {
    string_t first = strs.front();
    strs.erase( strs.begin() );
    return first;
} // shift

string_t
find(
    str_hash_t const & hash,
    string_t const &   key
) {
    string_t value;
    str_hash_t::const_iterator it = hash.find( key );
    if ( it != hash.end() ) {
        value = it->second;
    }; // if
    return value;
} // find

void die( string_t const & message ) {
    std::cerr << message << std::endl;
    exit( 1 );
} // die

void stop( string_t const & message ) {
    printf( "%s\n", message.c_str() );
    exit( 1 );
}

// An entry in the symbol table of a .obj file.
struct symbol_t {
    long long        name;
    unsigned         value;
    unsigned  short  section_num;
    unsigned  short  type;
    char             storage_class;
    char             nAux;
}; // struct symbol_t


class _rstream_t : public std::istrstream {

    private:

        const char * buf;

    protected:

        _rstream_t( pair< const char *, streamsize > p )
            : istrstream( p.first, p.second ), buf( p.first )
        {
        }

        ~_rstream_t() {
            delete [] buf;
        }

}; // class _rstream_t

/* A stream encapuslating the content of a file or the content of a string, overriding the
   >> operator to read various integer types in binary form, as well as a symbol table
   entry.
*/
class rstream_t : public _rstream_t {
private:

    template< typename type_t >
    inline rstream_t & do_read( type_t & x ) {
	read( (char*) & x, sizeof( type_t ) );
	return * this;
    }

    static pair<const char*, streamsize> getBuf(const char *fileName) {
	ifstream raw(fileName,ios::binary | ios::in);
	if(!raw.is_open())
	    stop("rstream.getBuf: Error opening file");
	raw.seekg(0,ios::end);
	streampos fileSize = raw.tellg();
	if(fileSize < 0)
	    stop("rstream.getBuf: Error reading file");
	char *buf = new char[fileSize];
	raw.seekg(0,ios::beg);
	raw.read(buf, fileSize);
	return pair<const char*, streamsize>(buf,fileSize);
    }
public:
    // construct from a string
    rstream_t( const char * buf, streamsize size ) :
        _rstream_t( pair< const char *, streamsize >( buf, size ) )
    {}
    /* construct from a file whole content is fully read once to initialize the content of
       this stream
    */
    rstream_t( string_t const & fileName )
        : _rstream_t( getBuf( fileName.c_str() ) )
    {
    }

    rstream_t & operator >>( int & x ) {
	return do_read(x);
    }
    rstream_t & operator >>(unsigned &x) {
	return do_read(x);
    }
    rstream_t & operator>>(short &x) {
	return do_read(x);
    }
    rstream_t & operator>>(unsigned short &x) {
	return do_read(x);
    }
    rstream_t & operator>>( symbol_t & e ) {
	read((char*)&e, 18);
	return *this;
    }
}; // class rstream_t

// string table in a .OBJ file
class StringTable {
private:
    map<string, unsigned> directory;
    size_t length;
    char *data;

    // make <directory> from <length> bytes in <data>
    void makeDirectory(void) {
	unsigned i = 4;
	while(i < length) {
	    string s = string(data + i);
	    directory.insert(make_pair(s, i));
	    i += s.size() + 1;
	}
    }
    // initialize <length> and <data> with contents specified by the arguments
    void init(const char *_data) {
	unsigned _length = *(unsigned*)_data;

	if(_length < sizeof(unsigned) || _length != *(unsigned*)_data)
	    stop("StringTable.init: Invalid symbol table");
	if(_data[_length - 1]) {
	    // to prevent runaway strings, make sure the data ends with a zero
	    data = new char[length = _length + 1];
	    data[_length] = 0;
	} else {
	    data = new char[length = _length];
	}
	*(unsigned*)data = length;
	memcpy( data + sizeof(unsigned), _data + sizeof(unsigned), length - sizeof(unsigned) );
	makeDirectory();
    }
public:
    StringTable( rstream_t & f ) {
	/* Construct string table by reading from f.
	 */
	streampos s;
	unsigned strSize;
	char *strData;

	s = f.tellg();
	f>>strSize;
	if(strSize < sizeof(unsigned))
	    stop("StringTable: Invalid string table");
	strData = new char[strSize];
	*(unsigned*)strData = strSize;
	// read the raw data into <strData>
	f.read(strData + sizeof(unsigned), strSize - sizeof(unsigned));
	s = f.tellg() - s;
	if(s < strSize)
	    stop("StringTable: Unexpected EOF");
	init(strData);
	delete[]strData;
    }
    StringTable(const set<string> &strings) {
	/* Construct string table from given strings.
	 */
	char *p;
	set<string>::const_iterator it;
	size_t s;

	// count required size for data
	for(length = sizeof(unsigned), it = strings.begin(); it != strings.end(); ++it) {
	    size_t l = (*it).size();

	    if(l > (unsigned) 0xFFFFFFFF)
		stop("StringTable: String too long");
	    if(l > 8) {
		length += l + 1;
		if(length > (unsigned) 0xFFFFFFFF)
		    stop("StringTable: Symbol table too long");
	    }
	}
	data = new char[length];
	*(unsigned*)data = length;
	// populate data and directory
	for(p = data + sizeof(unsigned), it = strings.begin(); it != strings.end(); ++it) {
	    const string &str = *it;
	    size_t l = str.size();
	    if(l > 8) {
		directory.insert(make_pair(str, p - data));
		memcpy(p, str.c_str(), l);
		p[l] = 0;
		p += l + 1;
	    }
	}
    }
    ~StringTable() {
	delete[] data;
    }
    /* Returns encoding for given string based on this string table.
       Error if string length is greater than 8 but string is not in
       the string table--returns 0.
    */
    int64_t encode(const string &str) {
	int64_t r;

	if(str.size() <= 8) {
	    // encoded directly
	    ((char*)&r)[7] = 0;
	    strncpy((char*)&r, str.c_str(), 8);
	    return r;
	} else {
	    // represented as index into table
	    map<string,unsigned>::const_iterator it = directory.find(str);
	    if(it == directory.end())
		stop("StringTable::encode: String now found in string table");
	    ((unsigned*)&r)[0] = 0;
	    ((unsigned*)&r)[1] = (*it).second;
	    return r;
	}
    }
    /* Returns string represented by x based on this string table.
       Error if x references an invalid position in the table--returns
       the empty string.
    */
    string decode(int64_t x) const {
	if(*(unsigned*)&x == 0) {
	    // represented as index into table
	    unsigned &p = ((unsigned*)&x)[1];
	    if(p >= length)
		stop("StringTable::decode: Invalid string table lookup");
	    return string(data + p);
	} else {
	    // encoded directly
	    char *p = (char*)&x;
	    int i;

	    for(i = 0; i < 8 && p[i]; ++i);
	    return string(p, i);
	}
    }
    void write(ostream &os) {
	os.write(data, length);
    }
};


void
obj_copy(
    string_t const &    src,    // Name of source file.
    string_t const &    dst,    // Name of destination file.
    str_hash_t const &  redefs  // List of redefinititions.
) {

    set< string > strings; // set of all occurring symbols, appropriately prefixed
    streampos fileSize;
    size_t strTabStart;
    unsigned symTabStart;
    unsigned symNEntries;
    int i;


    string const error_reading = "Error reading \"" + src + "\" file: ";

    rstream_t in( src );

    in.seekg( 0, ios::end );
    fileSize = in.tellg();

    in.seekg( 8 );
    in >> symTabStart >> symNEntries;
    strTabStart = symTabStart + 18 * size_t( symNEntries );
    in.seekg( strTabStart );
    if ( in.eof() ) {
        stop( error_reading + "Unexpected end of file" );
    }
    StringTable stringTableOld( in ); // Read original string table.

    if ( in.tellg() != fileSize ) {
        stop( error_reading + "Unexpected data after string table" );
    }

    // compute set of occurring strings with prefix added
    for ( i = 0; i < symNEntries; ++ i ) {

	symbol_t e;

	in.seekg( symTabStart + i * 18 );
	if ( in.eof() ) {
            stop("hideSymbols: Unexpected EOF");
        }
	in >> e;
	if ( in.fail() ) {
            stop("hideSymbols: File read error");
        }
	if ( e.nAux ) {
            i += e.nAux;
        }
	const string & s = stringTableOld.decode( e.name );
	// if symbol is extern and found in <hide>, prefix and insert into strings,
	// otherwise, just insert into strings without prefix
        string_t name = find( redefs, s );
	strings.insert( name != "" && e.storage_class == 2 ? name : s );
    }

    ofstream out( dst.c_str(), ios::trunc | ios::out | ios::binary );
    if ( ! out.is_open() ) {
        stop("hideSymbols: Error opening output file");
    }

    // make new string table from string set
    StringTable stringTableNew = StringTable( strings );

    // copy input file to output file up to just before the symbol table
    in.seekg( 0 );
    char * buf = new char[ symTabStart ];
    in.read( buf, symTabStart );
    out.write( buf, symTabStart );
    delete [] buf;

    // copy input symbol table to output symbol table with name translation
    for ( i = 0; i < symNEntries; ++ i ) {
	symbol_t e;

	in.seekg( symTabStart + i * 18 );
	if ( in.eof() ) {
            stop("hideSymbols: Unexpected EOF");
        }
	in >> e;
	if ( in.fail() ) {
            stop("hideSymbols: File read error");
        }
	const string & s = stringTableOld.decode( e.name );
	out.seekp( symTabStart + i * 18 );
        string_t name = find( redefs, s );
	e.name = stringTableNew.encode( ( e.storage_class == 2 && name != "" ) ? name : s );
	out.write( (char*) & e, 18 );
	if ( out.fail() ) {
            stop( "hideSymbols: File write error" );
        }
	if ( e.nAux ) {
	    // copy auxiliary symbol table entries
	    int nAux = e.nAux;
	    for (int j = 1; j <= nAux; ++j ) {
		in >> e;
		out.seekp( symTabStart + ( i + j ) * 18 );
		out.write( (char*) & e, 18 );
	    }
	    i += nAux;
	}
    }
    // output string table
    stringTableNew.write( out );
}


void
split( string_t const & str, char ch, string_t & head, string_t & tail ) {
    string_t::size_type pos = str.find( ch );
    if ( pos == string_t::npos ) {
        head = str;
        tail = "";
    } else {
        head = str.substr( 0, pos );
        tail = str.substr( pos + 1 );
    }; // if
} // split


void help() {
    std::cout
        << "NAME\n"
        << "    objcopy -- copy and translate object files\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    objcopy options... source destination\n"
        << "\n"
        << "OPTIONS\n"
        << "    --help  Print this help and exit.\n"
        << "    --redefine-sym old=new\n"
        << "            Rename \"old\" symbol in source object file to \"new\" symbol in\n"
        << "            destination object file.\n"
        << "    --redefine-syms sym_file\n"
        << "            For each pair \"old new\" in sym_file rename \"old\" symbol in \n"
        << "            source object file to \"new\" symbol in destination object file.\n"
        << "\n"
        << "ARGUMENTS\n"
        << "    source  The name of source object file.\n"
        << "    destination\n"
        << "            The name of destination object file.\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program implements a minor bit of Linux* OS's objcopy utility on Windows* OS.\n"
        << "    It can copy object files and edit its symbol table.\n"
        << "\n"
        << "EXAMPLES\n"
        << "    \n"
        << "        > objcopy --redefine-sym fastcpy=__xxx_fastcpy a.obj b.obj\n"
        << "\n";
} // help


int
main( int argc, char const * argv[] ) {

    strings_t   args( argc - 1 );
    str_hash_t  redefs;
    strings_t   files;

    std::copy( argv + 1, argv + argc, args.begin() );

    while ( args.size() > 0 ) {
        string_t arg = shift( args );
        if ( arg.substr( 0, 2 ) == "--" ) {
            // An option.
            if ( 0  ) {
            } else if ( arg == "--help" ) {
                help();
                return 0;
            } else if ( arg == "--redefine-sym" ) {
                if ( args.size() == 0 ) {
                    die( "\"" + arg + "\" option requires an argument" );
                }; // if
                // read list of symbol pairs "old new" from command line.
                string_t redef = shift( args );
                string_t old_sym;
                string_t new_sym;
                split( redef, '=', old_sym, new_sym );
                if ( old_sym.length() == 0 || new_sym.length() == 0 ) {
                    die( "Illegal redefinition: \"" + redef + "\"; neither old symbol nor new symbol may be empty" );
                }; // if
                redefs.insert( str_pair_t( old_sym, new_sym ) );
            } else if ( arg == "--redefine-syms" ) {
                if ( args.size() == 0 ) {
                    die( "\"" + arg + "\" option requires an argument" );
                }; // if
                // read list of symbol pairs "old new" from file.
                string_t fname = shift( args );
                string_t redef;
		ifstream ifs( fname.c_str() );
		while ( ifs.good() ) {
                    getline( ifs, redef );// get pair of old/new symbols separated by space
                    string_t old_sym;
                    string_t new_sym;
                    // AC: gcount() does not work here (always return 0), so comment it
                    //if ( ifs.gcount() ) { // skip empty lines
                    split( redef, ' ', old_sym, new_sym );
                    if ( old_sym.length() == 0 || new_sym.length() == 0 ) {
                        break;  // end of file reached (last empty line)
                        //die( "Illegal redefinition: \"" + redef + "\"; neither old symbol nor new symbol may be empty" );
                    }; // if
                    redefs.insert( str_pair_t( old_sym, new_sym ) );
                    //}
		}
            } else {
                die( "Illegal option: \"" + arg + "\"" );
            }; // if
        } else {
            // Not an option, a file name.
            if ( files.size() >= 2 ) {
                die( "Too many files specified; two files required (use --help option for help)" );
            }; // if
            files.push_back( arg );
        }; // if
    }; // while
    if ( files.size() < 2 ) {
        die( "Not enough files specified; two files required (use --help option for help)" );
    }; // if

    obj_copy( files[ 0 ], files[ 1 ], redefs );

    return 0;

} // main


// end of file //
