//===-- PythonDataObjectsTests.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Host/HostInfo.h"
#include "Plugins/ScriptInterpreter/Python/lldb-python.h"
#include "Plugins/ScriptInterpreter/Python/PythonDataObjects.h"
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"

using namespace lldb_private;

class PythonDataObjectsTest : public testing::Test
{
  public:
    void
    SetUp() override
    {
        HostInfoBase::Initialize();
        // ScriptInterpreterPython::Initialize() depends on things like HostInfo being initialized
        // so it can compute the python directory etc, so we need to do this after
        // SystemInitializerCommon::Initialize().
        ScriptInterpreterPython::Initialize();
    }

    void
    TearDown() override
    {
        ScriptInterpreterPython::Terminate();
    }
};

TEST_F(PythonDataObjectsTest, TestPythonInteger)
{
// Test that integers behave correctly when wrapped by a PythonInteger.

#if PY_MAJOR_VERSION < 3
    // Verify that `PythonInt` works correctly when given a PyInt object.
    // Note that PyInt doesn't exist in Python 3.x, so this is only for 2.x
    PyObject *py_int = PyInt_FromLong(12);
    EXPECT_TRUE(PythonInteger::Check(py_int));
    PythonInteger python_int(py_int);

    EXPECT_EQ(PyObjectType::Integer, python_int.GetObjectType());
    EXPECT_EQ(12, python_int.GetInteger());
#endif

    // Verify that `PythonInt` works correctly when given a PyLong object.
    PyObject *py_long = PyLong_FromLong(12);
    EXPECT_TRUE(PythonInteger::Check(py_long));
    PythonInteger python_long(py_long);
    EXPECT_EQ(PyObjectType::Integer, python_long.GetObjectType());

    // Verify that you can reset the value and that it is reflected properly.
    python_long.SetInteger(40);
    EXPECT_EQ(40, python_long.GetInteger());
}

TEST_F(PythonDataObjectsTest, TestPythonString)
{
    // Test that strings behave correctly when wrapped by a PythonString.

    static const char *test_string = "PythonDataObjectsTest::TestPythonString";
    static const char *test_string2 = "PythonDataObjectsTest::TestPythonString";

#if PY_MAJOR_VERSION < 3
    // Verify that `PythonString` works correctly when given a PyString object.
    // Note that PyString doesn't exist in Python 3.x, so this is only for 2.x
    PyObject *py_string = PyString_FromString(test_string);
    EXPECT_TRUE(PythonString::Check(py_string));
    PythonString python_string(py_string);

    EXPECT_EQ(PyObjectType::String, python_string.GetObjectType());
    EXPECT_STREQ(test_string, python_string.GetString().data());
#endif

    // Verify that `PythonString` works correctly when given a PyUnicode object.
    PyObject *py_unicode = PyUnicode_FromString(test_string);
    EXPECT_TRUE(PythonString::Check(py_unicode));
    PythonString python_unicode(py_unicode);

    EXPECT_EQ(PyObjectType::String, python_unicode.GetObjectType());
    EXPECT_STREQ(test_string, python_unicode.GetString().data());

    // Verify that you can reset the value and that it is reflected properly.
    python_unicode.SetString(test_string2);
    EXPECT_STREQ(test_string2, python_unicode.GetString().data());
}

TEST_F(PythonDataObjectsTest, TestPythonListPrebuilt)
{
    // Test that a list which is built through the native
    // Python API behaves correctly when wrapped by a PythonList.
    static const int list_size = 2;
    static const long long_idx0 = 5;
    static const char *const string_idx1 = "String Index 1";

    PyObject *list_items[list_size];

    PyObject *py_list = PyList_New(2);
    list_items[0] = PyLong_FromLong(long_idx0);
    list_items[1] = PyString_FromString(string_idx1);

    for (int i = 0; i < list_size; ++i)
        PyList_SetItem(py_list, i, list_items[i]);

    EXPECT_TRUE(PythonList::Check(py_list));

    PythonList list(py_list);
    EXPECT_EQ(list_size, list.GetSize());
    EXPECT_EQ(PyObjectType::List, list.GetObjectType());

    // PythonList doesn't yet support getting objects by type.
    // For now, we have to call CreateStructuredArray and use
    // those objects.  That will be in a different test.
    // TODO: Add the ability for GetItemByIndex() to return a
    // typed object.
}

TEST_F(PythonDataObjectsTest, TestPythonDictionaryPrebuilt)
{
    // Test that a dictionary which is built through the native
    // Python API behaves correctly when wrapped by a PythonDictionary.
    static const int dict_entries = 2;

    PyObject *keys[dict_entries];
    PyObject *values[dict_entries];

    keys[0] = PyString_FromString("Key 0");
    keys[1] = PyLong_FromLong(1);
    values[0] = PyLong_FromLong(0);
    values[1] = PyString_FromString("Value 1");

    PyObject *py_dict = PyDict_New();
    for (int i = 0; i < dict_entries; ++i)
        PyDict_SetItem(py_dict, keys[i], values[i]);

    EXPECT_TRUE(PythonDictionary::Check(py_dict));

    PythonDictionary dict(py_dict);
    EXPECT_EQ(dict.GetSize(), dict_entries);
    EXPECT_EQ(PyObjectType::Dictionary, dict.GetObjectType());

    // PythonDictionary doesn't yet support getting objects by type.
    // For now, we have to call CreateStructuredDictionary and use
    // those objects.  That will be in a different test.
    // TODO: Add the ability for GetItemByKey() to return a
    // typed object.
}

TEST_F(PythonDataObjectsTest, TestPythonListManipulation)
{
    // Test that manipulation of a PythonList behaves correctly when
    // wrapped by a PythonDictionary.

    static const long long_idx0 = 5;
    static const char *const string_idx1 = "String Index 1";

    PyObject *py_list = PyList_New(0);
    PythonList list(py_list);
    PythonInteger integer(long_idx0);
    PythonString string(string_idx1);

    list.AppendItem(integer);
    list.AppendItem(string);
    EXPECT_EQ(2, list.GetSize());

    // PythonList doesn't yet support getting typed objects out, so we
    // can't easily test that the first item is an integer with the correct
    // value, etc.
    // TODO: Add the ability for GetItemByIndex() to return a
    // typed object.
}

TEST_F(PythonDataObjectsTest, TestPythonDictionaryManipulation)
{
    // Test that manipulation of a dictionary behaves correctly when wrapped
    // by a PythonDictionary.
    static const int dict_entries = 2;

    PyObject *keys[dict_entries];
    PyObject *values[dict_entries];

    keys[0] = PyString_FromString("Key 0");
    keys[1] = PyString_FromString("Key 1");
    values[0] = PyLong_FromLong(1);
    values[1] = PyString_FromString("Value 1");

    PyObject *py_dict = PyDict_New();

    PythonDictionary dict(py_dict);
    for (int i = 0; i < 2; ++i)
        dict.SetItemForKey(PythonString(keys[i]), values[i]);

    EXPECT_EQ(dict_entries, dict.GetSize());

    // PythonDictionary doesn't yet support getting objects by type.
    // For now, we have to call CreateStructuredDictionary and use
    // those objects.  That will be in a different test.
    // TODO: Add the ability for GetItemByKey() to return a
    // typed object.
}

TEST_F(PythonDataObjectsTest, TestPythonListToStructuredObject)
{
    // Test that a PythonList is properly converted to a StructuredArray.
    // This includes verifying that a list can contain a nested list as
    // well as a nested dictionary.

    static const int item_count = 4;
    static const long long_idx0 = 5;
    static const char *const string_idx1 = "String Index 1";

    static const long nested_list_long_idx0 = 6;
    static const char *const nested_list_str_idx1 = "Nested String Index 1";

    static const char *const nested_dict_key0 = "Nested Key 0";
    static const char *const nested_dict_value0 = "Nested Value 0";
    static const char *const nested_dict_key1 = "Nested Key 1";
    static const long nested_dict_value1 = 2;

    PythonList list;
    PythonList nested_list;
    PythonDictionary nested_dict;

    nested_list.AppendItem(PythonInteger(nested_list_long_idx0));
    nested_list.AppendItem(PythonString(nested_list_str_idx1));
    nested_dict.SetItemForKey(PythonString(nested_dict_key0), PythonString(nested_dict_value0));
    nested_dict.SetItemForKey(PythonString(nested_dict_key1), PythonInteger(nested_dict_value1));

    list.AppendItem(PythonInteger(long_idx0));
    list.AppendItem(PythonString(string_idx1));
    list.AppendItem(nested_list);
    list.AppendItem(nested_dict);

    EXPECT_EQ(item_count, list.GetSize());

    StructuredData::ArraySP array_sp = list.CreateStructuredArray();
    EXPECT_EQ(list.GetSize(), array_sp->GetSize());
    EXPECT_EQ(StructuredData::Type::eTypeInteger, array_sp->GetItemAtIndex(0)->GetType());
    EXPECT_EQ(StructuredData::Type::eTypeString, array_sp->GetItemAtIndex(1)->GetType());
    EXPECT_EQ(StructuredData::Type::eTypeArray, array_sp->GetItemAtIndex(2)->GetType());
    EXPECT_EQ(StructuredData::Type::eTypeDictionary, array_sp->GetItemAtIndex(3)->GetType());

    auto list_int_sp = std::static_pointer_cast<StructuredData::Integer>(array_sp->GetItemAtIndex(0));
    auto list_str_sp = std::static_pointer_cast<StructuredData::String>(array_sp->GetItemAtIndex(1));
    auto list_list_sp = std::static_pointer_cast<StructuredData::Array>(array_sp->GetItemAtIndex(2));
    auto list_dict_sp = std::static_pointer_cast<StructuredData::Dictionary>(array_sp->GetItemAtIndex(3));

    // Verify that the first item (long) has the correct value
    EXPECT_EQ(long_idx0, list_int_sp->GetValue());

    // Verify that the second item (string) has the correct value
    EXPECT_STREQ(string_idx1, list_str_sp->GetValue().c_str());

    // Verify that the third item is a list with the correct length and element types
    EXPECT_EQ(nested_list.GetSize(), list_list_sp->GetSize());
    EXPECT_EQ(StructuredData::Type::eTypeInteger, list_list_sp->GetItemAtIndex(0)->GetType());
    EXPECT_EQ(StructuredData::Type::eTypeString, list_list_sp->GetItemAtIndex(1)->GetType());
    // Verify that the values of each element in the list are correct
    auto nested_list_value_0 = std::static_pointer_cast<StructuredData::Integer>(list_list_sp->GetItemAtIndex(0));
    auto nested_list_value_1 = std::static_pointer_cast<StructuredData::String>(list_list_sp->GetItemAtIndex(1));
    EXPECT_EQ(nested_list_long_idx0, nested_list_value_0->GetValue());
    EXPECT_STREQ(nested_list_str_idx1, nested_list_value_1->GetValue().c_str());

    // Verify that the fourth item is a dictionary with the correct length
    EXPECT_EQ(nested_dict.GetSize(), list_dict_sp->GetSize());
    auto dict_keys = std::static_pointer_cast<StructuredData::Array>(list_dict_sp->GetKeys());

    // Verify that all of the keys match the values and types of keys we inserted
    EXPECT_EQ(StructuredData::Type::eTypeString, dict_keys->GetItemAtIndex(0)->GetType());
    EXPECT_EQ(StructuredData::Type::eTypeString, dict_keys->GetItemAtIndex(1)->GetType());
    auto nested_key_0 = std::static_pointer_cast<StructuredData::String>(dict_keys->GetItemAtIndex(0));
    auto nested_key_1 = std::static_pointer_cast<StructuredData::String>(dict_keys->GetItemAtIndex(1));
    EXPECT_STREQ(nested_dict_key0, nested_key_0->GetValue().c_str());
    EXPECT_STREQ(nested_dict_key1, nested_key_1->GetValue().c_str());

    // Verify that for each key, the value has the correct type and value as what we inserted.
    auto nested_dict_value_0 = list_dict_sp->GetValueForKey(nested_key_0->GetValue());
    auto nested_dict_value_1 = list_dict_sp->GetValueForKey(nested_key_1->GetValue());
    EXPECT_EQ(StructuredData::Type::eTypeString, nested_dict_value_0->GetType());
    EXPECT_EQ(StructuredData::Type::eTypeInteger, nested_dict_value_1->GetType());
    auto nested_dict_str_value_0 = std::static_pointer_cast<StructuredData::String>(nested_dict_value_0);
    auto nested_dict_int_value_1 = std::static_pointer_cast<StructuredData::Integer>(nested_dict_value_1);
    EXPECT_STREQ(nested_dict_value0, nested_dict_str_value_0->GetValue().c_str());
    EXPECT_EQ(nested_dict_value1, nested_dict_int_value_1->GetValue());
}

TEST_F(PythonDataObjectsTest, TestPythonDictionaryToStructuredObject)
{
    // Test that a PythonDictionary is properly converted to a
    // StructuredDictionary.  This includes verifying that a dictionary
    // can contain a nested dictionary as well as a nested list.

    static const int dict_item_count = 4;
    static const char *const dict_keys[dict_item_count] = {"Key 0 (str)", "Key 1 (long)", "Key 2 (dict)",
                                                           "Key 3 (list)"};

    static const StructuredData::Type dict_value_types[dict_item_count] = {
        StructuredData::Type::eTypeString, StructuredData::Type::eTypeInteger, StructuredData::Type::eTypeDictionary,
        StructuredData::Type::eTypeArray};

    static const char *const nested_dict_keys[2] = {"Nested Key 0 (str)", "Nested Key 1 (long)"};

    static const StructuredData::Type nested_dict_value_types[2] = {
        StructuredData::Type::eTypeString, StructuredData::Type::eTypeInteger,
    };

    static const StructuredData::Type nested_list_value_types[2] = {StructuredData::Type::eTypeInteger,
                                                                    StructuredData::Type::eTypeString};

    static const char *const dict_value0 = "Value 0";
    static const long dict_value1 = 2;

    static const long nested_list_value0 = 5;
    static const char *const nested_list_value1 = "Nested list string";

    static const char *const nested_dict_value0 = "Nested Dict Value 0";
    static const long nested_dict_value1 = 7;

    PythonDictionary dict;
    PythonDictionary nested_dict;
    PythonList nested_list;

    nested_dict.SetItemForKey(PythonString(nested_dict_keys[0]), PythonString(nested_dict_value0));
    nested_dict.SetItemForKey(PythonString(nested_dict_keys[1]), PythonInteger(nested_dict_value1));

    nested_list.AppendItem(PythonInteger(nested_list_value0));
    nested_list.AppendItem(PythonString(nested_list_value1));

    dict.SetItemForKey(PythonString(dict_keys[0]), PythonString(dict_value0));
    dict.SetItemForKey(PythonString(dict_keys[1]), PythonInteger(dict_value1));
    dict.SetItemForKey(PythonString(dict_keys[2]), nested_dict);
    dict.SetItemForKey(PythonString(dict_keys[3]), nested_list);

    StructuredData::DictionarySP dict_sp = dict.CreateStructuredDictionary();
    EXPECT_EQ(dict_item_count, dict_sp->GetSize());
    auto dict_keys_array = std::static_pointer_cast<StructuredData::Array>(dict_sp->GetKeys());

    std::vector<StructuredData::StringSP> converted_keys;
    std::vector<StructuredData::ObjectSP> converted_values;
    // Verify that all of the keys match the values and types of keys we inserted
    // (Keys are always strings, so this is easy)
    for (int i = 0; i < dict_sp->GetSize(); ++i)
    {
        EXPECT_EQ(StructuredData::Type::eTypeString, dict_keys_array->GetItemAtIndex(i)->GetType());
        auto converted_key = std::static_pointer_cast<StructuredData::String>(dict_keys_array->GetItemAtIndex(i));
        converted_keys.push_back(converted_key);
        converted_values.push_back(dict_sp->GetValueForKey(converted_key->GetValue().c_str()));

        EXPECT_STREQ(dict_keys[i], converted_key->GetValue().c_str());
        EXPECT_EQ(dict_value_types[i], converted_values[i]->GetType());
    }

    auto dict_string_value = std::static_pointer_cast<StructuredData::String>(converted_values[0]);
    auto dict_int_value = std::static_pointer_cast<StructuredData::Integer>(converted_values[1]);
    auto dict_dict_value = std::static_pointer_cast<StructuredData::Dictionary>(converted_values[2]);
    auto dict_list_value = std::static_pointer_cast<StructuredData::Array>(converted_values[3]);

    // The first two dictionary values are easy to test, because they are just a string and an integer.
    EXPECT_STREQ(dict_value0, dict_string_value->GetValue().c_str());
    EXPECT_EQ(dict_value1, dict_int_value->GetValue());

    // For the nested dictionary, repeat the same process as before.
    EXPECT_EQ(2, dict_dict_value->GetSize());
    auto nested_dict_keys_array = std::static_pointer_cast<StructuredData::Array>(dict_dict_value->GetKeys());

    std::vector<StructuredData::StringSP> nested_converted_keys;
    std::vector<StructuredData::ObjectSP> nested_converted_values;
    // Verify that all of the keys match the values and types of keys we inserted
    // (Keys are always strings, so this is easy)
    for (int i = 0; i < dict_dict_value->GetSize(); ++i)
    {
        EXPECT_EQ(StructuredData::Type::eTypeString, nested_dict_keys_array->GetItemAtIndex(i)->GetType());
        auto converted_key =
            std::static_pointer_cast<StructuredData::String>(nested_dict_keys_array->GetItemAtIndex(i));
        nested_converted_keys.push_back(converted_key);
        nested_converted_values.push_back(dict_dict_value->GetValueForKey(converted_key->GetValue().c_str()));

        EXPECT_STREQ(nested_dict_keys[i], converted_key->GetValue().c_str());
        EXPECT_EQ(nested_dict_value_types[i], converted_values[i]->GetType());
    }

    auto converted_nested_dict_value_0 = std::static_pointer_cast<StructuredData::String>(nested_converted_values[0]);
    auto converted_nested_dict_value_1 = std::static_pointer_cast<StructuredData::Integer>(nested_converted_values[1]);

    // The first two dictionary values are easy to test, because they are just a string and an integer.
    EXPECT_STREQ(nested_dict_value0, converted_nested_dict_value_0->GetValue().c_str());
    EXPECT_EQ(nested_dict_value1, converted_nested_dict_value_1->GetValue());

    // For the nested list, just verify the size, type and value of each item
    nested_converted_values.clear();
    EXPECT_EQ(2, dict_list_value->GetSize());
    for (int i = 0; i < dict_list_value->GetSize(); ++i)
    {
        auto converted_value = dict_list_value->GetItemAtIndex(i);
        EXPECT_EQ(nested_list_value_types[i], converted_value->GetType());
        nested_converted_values.push_back(converted_value);
    }

    auto converted_nested_list_value_0 = std::static_pointer_cast<StructuredData::Integer>(nested_converted_values[0]);
    auto converted_nested_list_value_1 = std::static_pointer_cast<StructuredData::String>(nested_converted_values[1]);
    EXPECT_EQ(nested_list_value0, converted_nested_list_value_0->GetValue());
    EXPECT_STREQ(nested_list_value1, converted_nested_list_value_1->GetValue().c_str());
}
