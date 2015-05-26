//===-- XML.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/XML.h"

using namespace lldb;
using namespace lldb_private;


#pragma mark -- XMLDocument

XMLDocument::XMLDocument () :
    m_document (nullptr)
{
}

XMLDocument::~XMLDocument ()
{
    Clear();
}

void
XMLDocument::Clear()
{
#if defined( LIBXML2_DEFINED )
    if (m_document)
    {
        xmlDocPtr doc = m_document;
        m_document = nullptr;
        xmlFreeDoc(doc);
    }
#endif
}

bool
XMLDocument::IsValid() const
{
    return m_document != nullptr;
}

void
XMLDocument::ErrorCallback (void *ctx, const char *format, ...)
{
    XMLDocument *document = (XMLDocument *)ctx;
    va_list args;
    va_start (args, format);
    document->m_errors.PrintfVarArg(format, args);
    document->m_errors.EOL();
    va_end (args);
}

bool
XMLDocument::ParseFile (const char *path)
{
#if defined( LIBXML2_DEFINED )
    Clear();
    xmlSetGenericErrorFunc( (void *)this, XMLDocument::ErrorCallback );
    m_document = xmlParseFile(path);
    xmlSetGenericErrorFunc(nullptr, nullptr);
#endif
    return IsValid();
}

bool
XMLDocument::ParseMemory (const char *xml, size_t xml_length, const char *url)
{
#if defined( LIBXML2_DEFINED )
    Clear();
    xmlSetGenericErrorFunc( (void *)this, XMLDocument::ErrorCallback );
    m_document = xmlReadMemory(xml, (int)xml_length, url, nullptr, 0);
    xmlSetGenericErrorFunc(nullptr, nullptr);
#endif
    return IsValid();
    
}

XMLNode
XMLDocument::GetRootElement(const char *required_name)
{
#if defined( LIBXML2_DEFINED )
    if (IsValid())
    {
        XMLNode root_node(xmlDocGetRootElement(m_document));
        if (required_name)
        {
            llvm::StringRef actual_name = root_node.GetName();
            if (actual_name == required_name)
                return root_node;
        }
        else
        {
            return root_node;
        }
    }
#endif
    return XMLNode();
}

bool
XMLDocument::XMLEnabled ()
{
#if defined( LIBXML2_DEFINED )
    return true;
#else
    return false;
#endif
}

#pragma mark -- XMLNode

XMLNode::XMLNode() :
    m_node(nullptr)
{
}

XMLNode::XMLNode(XMLNodeImpl node) :
    m_node(node)
{
}

XMLNode::~XMLNode()
{
    
}

void
XMLNode::Clear()
{
    m_node = nullptr;
}

XMLNode
XMLNode::GetParent() const
{
#if defined( LIBXML2_DEFINED )
    if (IsValid())
        return XMLNode(m_node->parent);
    else
        return XMLNode();
#else
    return XMLNode();
#endif

}

XMLNode
XMLNode::GetSibling() const
{
#if defined( LIBXML2_DEFINED )
    if (IsValid())
        return XMLNode(m_node->next);
    else
        return XMLNode();
#else
    return XMLNode();
#endif

}

XMLNode
XMLNode::GetChild () const
{
#if defined( LIBXML2_DEFINED )

    if (IsValid())
        return XMLNode(m_node->children);
    else
        return XMLNode();
#else
    return XMLNode();
#endif

}

llvm::StringRef
XMLNode::GetAttributeValue(const char *name, const char *fail_value) const
{
    const char *attr_value = NULL;
#if defined( LIBXML2_DEFINED )

    if (IsValid())
        attr_value = (const char *)xmlGetProp(m_node, (const xmlChar *)name);
    else
        attr_value = fail_value;
#else
    attr_value = fail_value;
#endif
    if (attr_value)
        return llvm::StringRef(attr_value);
    else
        return llvm::StringRef();
}




void
XMLNode::ForEachChildNode (NodeCallback const &callback) const
{
#if defined( LIBXML2_DEFINED )
    if (IsValid())
        GetChild().ForEachSiblingNode(callback);
#endif
}

void
XMLNode::ForEachChildElement (NodeCallback const &callback) const
{
#if defined( LIBXML2_DEFINED )
    XMLNode child = GetChild();
    if (child)
        child.ForEachSiblingElement(callback);
#endif
}

void
XMLNode::ForEachChildElementWithName (const char *name, NodeCallback const &callback) const
{
#if defined( LIBXML2_DEFINED )
    XMLNode child = GetChild();
    if (child)
        child.ForEachSiblingElementWithName(name, callback);
#endif
}

void
XMLNode::ForEachAttribute (AttributeCallback const &callback) const
{
#if defined( LIBXML2_DEFINED )

    if (IsValid())
    {
        for (xmlAttrPtr attr = m_node->properties; attr != nullptr; attr=attr->next)
        {
            // check if name matches
            if (attr->name)
            {
                // check child is a text node
                xmlNodePtr child = attr->children;
                if (child->type == XML_TEXT_NODE)
                {
                    llvm::StringRef attr_value;
                    if (child->content)
                        attr_value = llvm::StringRef((const char *)child->content);
                    if (callback(llvm::StringRef((const char *)attr->name), attr_value) == false)
                        return;
                }
            }
        }
    }
#endif
}


void
XMLNode::ForEachSiblingNode (NodeCallback const &callback) const
{
#if defined( LIBXML2_DEFINED )

    if (IsValid())
    {
        // iterate through all siblings
        for (xmlNodePtr node = m_node; node; node=node->next)
        {
            if (callback(XMLNode(node)) == false)
                return;
        }
    }
#endif
}

void
XMLNode::ForEachSiblingElement (NodeCallback const &callback) const
{
#if defined( LIBXML2_DEFINED )
    
    if (IsValid())
    {
        // iterate through all siblings
        for (xmlNodePtr node = m_node; node; node=node->next)
        {
            // we are looking for element nodes only
            if (node->type != XML_ELEMENT_NODE)
                continue;
            
            if (callback(XMLNode(node)) == false)
                return;
        }
    }
#endif
}

void
XMLNode::ForEachSiblingElementWithName (const char *name, NodeCallback const &callback) const
{
#if defined( LIBXML2_DEFINED )
    
    if (IsValid())
    {
        // iterate through all siblings
        for (xmlNodePtr node = m_node; node; node=node->next)
        {
            // we are looking for element nodes only
            if (node->type != XML_ELEMENT_NODE)
                continue;
            
            // If name is nullptr, we take all nodes of type "t", else
            // just the ones whose name matches
            if (name)
            {
                if (strcmp((const char *)node->name, name) != 0)
                    continue; // Name mismatch, ignore this one
            }
            else
            {
                if (node->name)
                    continue; // nullptr name specified and this elemnt has a name, ignore this one
            }
            
            if (callback(XMLNode(node)) == false)
                return;
        }
    }
#endif
}

llvm::StringRef
XMLNode::GetName() const
{
#if defined( LIBXML2_DEFINED )
    if (IsValid())
    {
        if (m_node->name)
            return llvm::StringRef((const char *)m_node->name);
    }
#endif
    return llvm::StringRef();
}

bool
XMLNode::GetElementText (std::string &text) const
{
    text.clear();
#if defined( LIBXML2_DEFINED )
    if (IsValid())
    {
        bool success = false;
        if (m_node->type == XML_ELEMENT_NODE)
        {
            // check child is a text node
            for (xmlNodePtr node = m_node->children;
                 node != nullptr;
                 node = node->next)
            {
                if (node->type == XML_TEXT_NODE)
                {
                    text.append((const char *)node->content);
                    success = true;
                }
            }
        }
        return success;
    }
#endif
    return false;
}



bool
XMLNode::NameIs (const char *name) const
{
#if defined( LIBXML2_DEFINED )

    if (IsValid())
    {
        // In case we are looking for a nullptr name or an exact pointer match
        if (m_node->name == (const xmlChar *)name)
            return true;
        if (m_node->name)
            return strcmp((const char *)m_node->name, name) == 0;
    }
#endif
    return false;
}

XMLNode
XMLNode::FindFirstChildElementWithName (const char *name) const
{
    XMLNode result_node;

#if defined( LIBXML2_DEFINED )
    ForEachChildElementWithName(name, [&result_node, name](const XMLNode& node) -> bool {
        result_node = node;
        // Stop iterating, we found the node we wanted
        return false;
    });
#endif

    return result_node;
}

bool
XMLNode::IsValid() const
{
    return m_node != nullptr;
}

bool
XMLNode::IsElement () const
{
#if defined( LIBXML2_DEFINED )
    if (IsValid())
        return m_node->type == XML_ELEMENT_NODE;
#endif
    return false;
}


XMLNode
XMLNode::GetElementForPath (const NamePath &path)
{
#if defined( LIBXML2_DEFINED )

    if (IsValid())
    {
        if (path.empty())
            return *this;
        else
        {
            XMLNode node = FindFirstChildElementWithName(path[0].c_str());
            const size_t n = path.size();
            for (size_t i=1; node && i<n; ++i)
                node = node.FindFirstChildElementWithName(path[i].c_str());
            return node;
        }
    }
#endif

    return XMLNode();
}


#pragma mark -- ApplePropertyList

ApplePropertyList::ApplePropertyList() :
    m_xml_doc(),
    m_dict_node()
{
    
}

ApplePropertyList::ApplePropertyList (const char *path) :
    m_xml_doc(),
    m_dict_node()
{
    ParseFile(path);
}

bool
ApplePropertyList::ParseFile (const char *path)
{
    if (m_xml_doc.ParseFile(path))
    {
        XMLNode plist = m_xml_doc.GetRootElement("plist");
        if (plist)
        {
            plist.ForEachChildElementWithName("dict", [this](const XMLNode &dict) -> bool {
                this->m_dict_node = dict;
                return false; // Stop iterating
            });
            return (bool)m_dict_node;
        }
    }
    return false;
}

bool
ApplePropertyList::IsValid() const
{
    return (bool)m_dict_node;
}

bool
ApplePropertyList::GetValueAsString (const char *key, std::string &value) const
{
    XMLNode value_node = GetValueNode (key);
    if (value_node)
        return ApplePropertyList::ExtractStringFromValueNode(value_node, value);
    return false;
}

XMLNode
ApplePropertyList::GetValueNode (const char *key) const
{
    XMLNode value_node;
#if defined( LIBXML2_DEFINED )
    
    if (IsValid())
    {
        m_dict_node.ForEachChildElementWithName("key", [key, &value_node](const XMLNode &key_node) -> bool {
            std::string key_name;
            if (key_node.GetElementText(key_name))
            {
                if (key_name.compare(key) == 0)
                {
                    value_node = key_node.GetSibling();
                    while (value_node && !value_node.IsElement())
                        value_node = value_node.GetSibling();
                    return false; // Stop iterating
                }
            }
            return true; // Keep iterating
        });
    }
#endif
    return value_node;
}

bool
ApplePropertyList::ExtractStringFromValueNode (const XMLNode &node, std::string &value)
{
    value.clear();
#if defined( LIBXML2_DEFINED )
    if (node.IsValid())
    {
        llvm::StringRef element_name = node.GetName();
        if (element_name == "true" or element_name == "false")
        {
            // The text value _is_ the element name itself...
            value = std::move(element_name.str());
            return true;
        }
        else if (element_name == "dict" or element_name == "array")
            return false; // dictionaries and arrays have no text value, so we fail
        else
            return node.GetElementText(value);
    }
#endif
    return false;
}

