//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


/*! \file
    \brief Function and Variable tables used by the runtime library
*/

#ifndef OFFLOAD_TABLE_H_INCLUDED
#define OFFLOAD_TABLE_H_INCLUDED

#include <iterator>
#include "offload_util.h"

// Template representing double linked list of tables
template <typename T> class TableList {
public:
    // table type
    typedef T Table;

    // List node
    struct Node {
        Table   table;
        Node*   prev;
        Node*   next;
    };

public:
    explicit TableList(Node *node = 0) : m_head(node) {}

    void add_table(Node *node) {
        m_lock.lock();

        if (m_head != 0) {
            node->next = m_head;
            m_head->prev = node;
        }
        m_head = node;

        m_lock.unlock();
    }

    void remove_table(Node *node) {
        m_lock.lock();

        if (node->next != 0) {
            node->next->prev = node->prev;
        }
        if (node->prev != 0) {
            node->prev->next = node->next;
        }
        if (m_head == node) {
            m_head = node->next;
        }

        m_lock.unlock();
    }

protected:
    Node*           m_head;
    mutex_t         m_lock;
};

// Function lookup table.
struct FuncTable {
    //! Function table entry
    /*! This table contains functions created from offload regions.   */
    /*! Each entry consists of a pointer to the function's "key"
        and the function address.                                     */
    /*! Each shared library or executable may contain one such table. */
    /*! The end of the table is marked with an entry whose name field
        has value -1.                                                 */
    struct Entry {
        const char* name; //!< Name of the function
        void*       func; //!< Address of the function
    };

    // entries
    const Entry *entries;

    // max name length
    int64_t max_name_len;
};

// Function table
class FuncList : public TableList<FuncTable> {
public:
    explicit FuncList(Node *node = 0) : TableList<Table>(node),
                                        m_max_name_len(-1)
    {}

    // add table to the list
    void add_table(Node *node) {
        // recalculate max function name length
        m_max_name_len = -1;

        // add table
        TableList<Table>::add_table(node);
    }

    // find function address for the given name
    const void* find_addr(const char *name);

    // find function name for the given address
    const char* find_name(const void *addr);

    // max name length from all tables in the list
    int64_t max_name_length(void);

    // debug dump
    void dump(void);

private:
    // max name length within from all tables
    int64_t m_max_name_len;
};

// Table entry for static variables
struct VarTable {
    //! Variable table entry
    /*! This table contains statically allocated variables marked with
        __declspec(target(mic) or #pragma omp declare target.           */
    /*! Each entry consists of a pointer to the variable's "key",
        the variable address and its size in bytes.                     */
    /*! Because memory allocation is done from the host,
        the MIC table does not need the size of the variable.           */
    /*! Padding to make the table entry size a power of 2 is necessary
        to avoid "holes" between table contributions from different object
        files on Windows when debug information is specified with /Zi.  */
    struct Entry {
        const char* name; //!< Name of the variable
        void*       addr; //!< Address of the variable

#if HOST_LIBRARY
        uint64_t    size;

#ifdef TARGET_WINNT
		// padding to make entry size a power of 2
        uint64_t    padding;
#endif // TARGET_WINNT
#endif
    };

    // Table terminated by an entry with name == -1
    const Entry *entries;
};

// List of var tables
class VarList : public TableList<VarTable> {
public:
    VarList() : TableList<Table>()
    {}

    // debug dump
    void dump();

public:
    // var table list iterator
    class Iterator : public std::iterator<std::input_iterator_tag,
                                          Table::Entry> {
    public:
        Iterator() : m_node(0), m_entry(0) {}

        explicit Iterator(Node *node) {
            new_node(node);
        }

        Iterator& operator++() {
            if (m_entry != 0) {
                m_entry++;
                while (m_entry->name == 0) {
                    m_entry++;
                }
                if (m_entry->name == reinterpret_cast<const char*>(-1)) {
                    new_node(m_node->next);
                }
            }
            return *this;
        }

        bool operator==(const Iterator &other) const {
            return m_entry == other.m_entry;
        }

        bool operator!=(const Iterator &other) const {
            return m_entry != other.m_entry;
        }

        const Table::Entry* operator*() const {
            return m_entry;
        }

    private:
        void new_node(Node *node) {
            m_node = node;
            m_entry = 0;
            while (m_node != 0) {
                m_entry = m_node->table.entries;
                while (m_entry->name == 0) {
                    m_entry++;
                }
                if (m_entry->name != reinterpret_cast<const char*>(-1)) {
                    break;
                }
                m_node = m_node->next;
                m_entry = 0;
            }
        }

    private:
        Node                *m_node;
        const Table::Entry  *m_entry;
    };

    Iterator begin() const {
        return Iterator(m_head);
    }

    Iterator end() const {
        return Iterator();
    }

public:
    // Entry representation in a copy buffer
    struct BufEntry {
        intptr_t name;
        intptr_t addr;
    };

    // Calculate the number of elements in the table and
    // returns the size of buffer for the table
    int64_t table_size(int64_t &nelems);

    // Copy table contents to given buffer. It is supposed to be large
    // enough to hold all elements as string table.
    void table_copy(void *buf, int64_t nelems);

    // Patch name offsets in a table after it's been copied to other side
    static void table_patch_names(void *buf, int64_t nelems);
};

extern FuncList __offload_entries;
extern FuncList __offload_funcs;
extern VarList  __offload_vars;

// Section names where the lookup tables are stored
#ifdef TARGET_WINNT
#define OFFLOAD_ENTRY_TABLE_SECTION_START   ".OffloadEntryTable$a"
#define OFFLOAD_ENTRY_TABLE_SECTION_END     ".OffloadEntryTable$z"

#define OFFLOAD_FUNC_TABLE_SECTION_START    ".OffloadFuncTable$a"
#define OFFLOAD_FUNC_TABLE_SECTION_END      ".OffloadFuncTable$z"

#define OFFLOAD_VAR_TABLE_SECTION_START     ".OffloadVarTable$a"
#define OFFLOAD_VAR_TABLE_SECTION_END       ".OffloadVarTable$z"

#define OFFLOAD_CRTINIT_SECTION_START       ".CRT$XCT"

#pragma section(OFFLOAD_CRTINIT_SECTION_START, read)

#else  // TARGET_WINNT

#define OFFLOAD_ENTRY_TABLE_SECTION_START   ".OffloadEntryTable."
#define OFFLOAD_ENTRY_TABLE_SECTION_END     ".OffloadEntryTable."

#define OFFLOAD_FUNC_TABLE_SECTION_START    ".OffloadFuncTable."
#define OFFLOAD_FUNC_TABLE_SECTION_END      ".OffloadFuncTable."

#define OFFLOAD_VAR_TABLE_SECTION_START     ".OffloadVarTable."
#define OFFLOAD_VAR_TABLE_SECTION_END       ".OffloadVarTable."
#endif // TARGET_WINNT

#pragma section(OFFLOAD_ENTRY_TABLE_SECTION_START, read, write)
#pragma section(OFFLOAD_ENTRY_TABLE_SECTION_END, read, write)

#pragma section(OFFLOAD_FUNC_TABLE_SECTION_START, read, write)
#pragma section(OFFLOAD_FUNC_TABLE_SECTION_END, read, write)

#pragma section(OFFLOAD_VAR_TABLE_SECTION_START, read, write)
#pragma section(OFFLOAD_VAR_TABLE_SECTION_END, read, write)


// register/unregister given tables
extern "C" void __offload_register_tables(
    FuncList::Node *entry_table,
    FuncList::Node *func_table,
    VarList::Node *var_table
);

extern "C" void __offload_unregister_tables(
    FuncList::Node *entry_table,
    FuncList::Node *func_table,
    VarList::Node *var_table
);
#endif  // OFFLOAD_TABLE_H_INCLUDED
