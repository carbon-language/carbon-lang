//===--Graph.cpp--- implements Graph class ---------------- ------*- C++ -*--=//
//
// This implements Graph for helping in trace generation
// This graph gets used by "ProfilePaths" class
//
//===----------------------------------------------------------------------===//

#include "Graph.h"
#include "llvm/iTerminators.h"
#include "Support/Debug.h"
#include <algorithm>

using std::map;
using std::vector;
using std::cerr;

const graphListElement *findNodeInList(const Graph::nodeList &NL,
					      Node *N) {
  for(Graph::nodeList::const_iterator NI = NL.begin(), NE=NL.end(); NI != NE; 
      ++NI)
    if (*NI->element== *N)
      return &*NI;
  return 0;
}

graphListElement *findNodeInList(Graph::nodeList &NL, Node *N) {
  for(Graph::nodeList::iterator NI = NL.begin(), NE=NL.end(); NI != NE; ++NI)
    if (*NI->element== *N)
      return &*NI;
  return 0;
}

//graph constructor with root and exit specified
Graph::Graph(std::vector<Node*> n, std::vector<Edge> e, 
	     Node *rt, Node *lt){
  strt=rt;
  ext=lt;
  for(vector<Node* >::iterator x=n.begin(), en=n.end(); x!=en; ++x)
    //nodes[*x] = list<graphListElement>();
    nodes[*x] = vector<graphListElement>();

  for(vector<Edge >::iterator x=e.begin(), en=e.end(); x!=en; ++x){
    Edge ee=*x;
    int w=ee.getWeight();
    //nodes[ee.getFirst()].push_front(graphListElement(ee.getSecond(),w, ee.getRandId()));   
    nodes[ee.getFirst()].push_back(graphListElement(ee.getSecond(),w, ee.getRandId()));
  }
  
}

//sorting edgelist, called by backEdgeVist ONLY!!!
Graph::nodeList &Graph::sortNodeList(Node *par, nodeList &nl, vector<Edge> &be){
  assert(par && "null node pointer");
  BasicBlock *bbPar = par->getElement();
  
  if(nl.size()<=1) return nl;
  if(getExit() == par) return nl;

  for(nodeList::iterator NLI = nl.begin(), NLE = nl.end()-1; NLI != NLE; ++NLI){
    nodeList::iterator min = NLI;
    for(nodeList::iterator LI = NLI+1, LE = nl.end(); LI!=LE; ++LI){
      //if LI < min, min = LI
      if(min->element->getElement() == LI->element->getElement() &&
         min->element == getExit()){

        //same successors: so might be exit???
        //if it is exit, then see which is backedge
        //check if LI is a left back edge!

        TerminatorInst *tti = par->getElement()->getTerminator();
        BranchInst *ti =  cast<BranchInst>(tti);

        assert(ti && "not a branch");
        assert(ti->getNumSuccessors()==2 && "less successors!");
        
        BasicBlock *tB = ti->getSuccessor(0);
        BasicBlock *fB = ti->getSuccessor(1);
        //so one of LI or min must be back edge!
        //Algo: if succ(0)!=LI (and so !=min) then succ(0) is backedge
        //and then see which of min or LI is backedge
        //THEN if LI is in be, then min=LI
        if(LI->element->getElement() != tB){//so backedge must be made min!
          for(vector<Edge>::iterator VBEI = be.begin(), VBEE = be.end();
              VBEI != VBEE; ++VBEI){
            if(VBEI->getRandId() == LI->randId){
              min = LI;
              break;
            }
            else if(VBEI->getRandId() == min->randId)
              break;
          }
        }
        else{// if(LI->element->getElement() != fB)
          for(vector<Edge>::iterator VBEI = be.begin(), VBEE = be.end();
              VBEI != VBEE; ++VBEI){
            if(VBEI->getRandId() == min->randId){
              min = LI;
              break;
            }
            else if(VBEI->getRandId() == LI->randId)
              break;
          }
        }
      }
      
      else if (min->element->getElement() != LI->element->getElement()){
        TerminatorInst *tti = par->getElement()->getTerminator();
        BranchInst *ti =  cast<BranchInst>(tti);
        assert(ti && "not a branch");

        if(ti->getNumSuccessors()<=1) continue;
        
        assert(ti->getNumSuccessors()==2 && "less successors!");
        
        BasicBlock *tB = ti->getSuccessor(0);
        BasicBlock *fB = ti->getSuccessor(1);
        
        if(tB == LI->element->getElement() || fB == min->element->getElement())
          min = LI;
      }
    }
    
    graphListElement tmpElmnt = *min;
    *min = *NLI;
    *NLI = tmpElmnt;
  }
  return nl;
}

//check whether graph has an edge
//having an edge simply means that there is an edge in the graph
//which has same endpoints as the given edge
bool Graph::hasEdge(Edge ed){
  if(ed.isNull())
    return false;

  nodeList &nli= nodes[ed.getFirst()]; //getNodeList(ed.getFirst());
  Node *nd2=ed.getSecond();

  return (findNodeInList(nli,nd2)!=NULL);

}


//check whether graph has an edge, with a given wt
//having an edge simply means that there is an edge in the graph
//which has same endpoints as the given edge
//This function checks, moreover, that the wt of edge matches too
bool Graph::hasEdgeAndWt(Edge ed){
  if(ed.isNull())
    return false;

  Node *nd2=ed.getSecond();
  nodeList &nli = nodes[ed.getFirst()];//getNodeList(ed.getFirst());
  
  for(nodeList::iterator NI=nli.begin(), NE=nli.end(); NI!=NE; ++NI)
    if(*NI->element == *nd2 && ed.getWeight()==NI->weight)
      return true;
  
  return false;
}

//add a node
void Graph::addNode(Node *nd){
  vector<Node *> lt=getAllNodes();

  for(vector<Node *>::iterator LI=lt.begin(), LE=lt.end(); LI!=LE;++LI){
    if(**LI==*nd)
      return;
  }
  //chng
  nodes[nd] =vector<graphListElement>(); //list<graphListElement>();
}

//add an edge
//this adds an edge ONLY when 
//the edge to be added does not already exist
//we "equate" two edges here only with their 
//end points
void Graph::addEdge(Edge ed, int w){
  nodeList &ndList = nodes[ed.getFirst()];
  Node *nd2=ed.getSecond();

  if(findNodeInList(nodes[ed.getFirst()], nd2))
    return;
 
  //ndList.push_front(graphListElement(nd2,w, ed.getRandId()));
  ndList.push_back(graphListElement(nd2,w, ed.getRandId()));//chng
  //sortNodeList(ed.getFirst(), ndList);

  //sort(ndList.begin(), ndList.end(), NodeListSort());
}

//add an edge EVEN IF such an edge already exists
//this may make a multi-graph
//which does happen when we add dummy edges
//to the graph, for compensating for back-edges
void Graph::addEdgeForce(Edge ed){
  //nodes[ed.getFirst()].push_front(graphListElement(ed.getSecond(),
  //ed.getWeight(), ed.getRandId()));
  nodes[ed.getFirst()].push_back
    (graphListElement(ed.getSecond(), ed.getWeight(), ed.getRandId()));

  //sortNodeList(ed.getFirst(), nodes[ed.getFirst()]);
  //sort(nodes[ed.getFirst()].begin(), nodes[ed.getFirst()].end(), NodeListSort());
}

//remove an edge
//Note that it removes just one edge,
//the first edge that is encountered
void Graph::removeEdge(Edge ed){
  nodeList &ndList = nodes[ed.getFirst()];
  Node &nd2 = *ed.getSecond();

  for(nodeList::iterator NI=ndList.begin(), NE=ndList.end(); NI!=NE ;++NI) {
    if(*NI->element == nd2) {
      ndList.erase(NI);
      break;
    }
  }
}

//remove an edge with a given wt
//Note that it removes just one edge,
//the first edge that is encountered
void Graph::removeEdgeWithWt(Edge ed){
  nodeList &ndList = nodes[ed.getFirst()];
  Node &nd2 = *ed.getSecond();

  for(nodeList::iterator NI=ndList.begin(), NE=ndList.end(); NI!=NE ;++NI) {
    if(*NI->element == nd2 && NI->weight==ed.getWeight()) {
      ndList.erase(NI);
      break;
    }
  }
}

//set the weight of an edge
void Graph::setWeight(Edge ed){
  graphListElement *El = findNodeInList(nodes[ed.getFirst()], ed.getSecond());
  if (El)
    El->weight=ed.getWeight();
}



//get the list of successor nodes
vector<Node *> Graph::getSuccNodes(Node *nd){
  nodeMapTy::const_iterator nli = nodes.find(nd);
  assert(nli != nodes.end() && "Node must be in nodes map");
  const nodeList &nl = getNodeList(nd);//getSortedNodeList(nd);

  vector<Node *> lt;
  for(nodeList::const_iterator NI=nl.begin(), NE=nl.end(); NI!=NE; ++NI)
    lt.push_back(NI->element);

  return lt;
}

//get the number of outgoing edges
int Graph::getNumberOfOutgoingEdges(Node *nd) const {
  nodeMapTy::const_iterator nli = nodes.find(nd);
  assert(nli != nodes.end() && "Node must be in nodes map");
  const nodeList &nl = nli->second;

  int count=0;
  for(nodeList::const_iterator NI=nl.begin(), NE=nl.end(); NI!=NE; ++NI)
    count++;

  return count;
}

//get the list of predecessor nodes
vector<Node *> Graph::getPredNodes(Node *nd){
  vector<Node *> lt;
  for(nodeMapTy::const_iterator EI=nodes.begin(), EE=nodes.end(); EI!=EE ;++EI){
    Node *lnode=EI->first;
    const nodeList &nl = getNodeList(lnode);

    const graphListElement *N = findNodeInList(nl, nd);
    if (N) lt.push_back(lnode);
  }
  return lt;
}

//get the number of predecessor nodes
int Graph::getNumberOfIncomingEdges(Node *nd){
  int count=0;
  for(nodeMapTy::const_iterator EI=nodes.begin(), EE=nodes.end(); EI!=EE ;++EI){
    Node *lnode=EI->first;
    const nodeList &nl = getNodeList(lnode);
    for(Graph::nodeList::const_iterator NI = nl.begin(), NE=nl.end(); NI != NE; 
	++NI)
      if (*NI->element== *nd)
	count++;
  }
  return count;
}

//get the list of all the vertices in graph
vector<Node *> Graph::getAllNodes() const{
  vector<Node *> lt;
  for(nodeMapTy::const_iterator x=nodes.begin(), en=nodes.end(); x != en; ++x)
    lt.push_back(x->first);

  return lt;
}

//get the list of all the vertices in graph
vector<Node *> Graph::getAllNodes(){
  vector<Node *> lt;
  for(nodeMapTy::const_iterator x=nodes.begin(), en=nodes.end(); x != en; ++x)
    lt.push_back(x->first);

  return lt;
}

//class to compare two nodes in graph
//based on their wt: this is used in
//finding the maximal spanning tree
struct compare_nodes {
  bool operator()(Node *n1, Node *n2){
    return n1->getWeight() < n2->getWeight();
  }
};


static void printNode(Node *nd){
  cerr<<"Node:"<<nd->getElement()->getName()<<"\n";
}

//Get the Maximal spanning tree (also a graph)
//of the graph
Graph* Graph::getMaxSpanningTree(){
  //assume connected graph
 
  Graph *st=new Graph();//max spanning tree, undirected edges
  int inf=9999999;//largest key
  vector<Node *> lt = getAllNodes();
  
  //initially put all vertices in vector vt
  //assign wt(root)=0
  //wt(others)=infinity
  //
  //now:
  //pull out u: a vertex frm vt of min wt
  //for all vertices w in vt, 
  //if wt(w) greater than 
  //the wt(u->w), then assign
  //wt(w) to be wt(u->w).
  //
  //make parent(u)=w in the spanning tree
  //keep pulling out vertices from vt till it is empty

  vector<Node *> vt;
  
  map<Node*, Node* > parent;
  map<Node*, int > ed_weight;

  //initialize: wt(root)=0, wt(others)=infinity
  //parent(root)=NULL, parent(others) not defined (but not null)
  for(vector<Node *>::iterator LI=lt.begin(), LE=lt.end(); LI!=LE; ++LI){
    Node *thisNode=*LI;
    if(*thisNode == *getRoot()){
      thisNode->setWeight(0);
      parent[thisNode]=NULL;
      ed_weight[thisNode]=0;
    }
    else{ 
      thisNode->setWeight(inf);
    }
    st->addNode(thisNode);//add all nodes to spanning tree
    //we later need to assign edges in the tree
    vt.push_back(thisNode); //pushed all nodes in vt
  }

  //keep pulling out vertex of min wt from vt
  while(!vt.empty()){
    Node *u=*(min_element(vt.begin(), vt.end(), compare_nodes()));
    DEBUG(cerr<<"popped wt"<<(u)->getWeight()<<"\n";
          printNode(u));

    if(parent[u]!=NULL){ //so not root
      Edge edge(parent[u],u, ed_weight[u]); //assign edge in spanning tree
      st->addEdge(edge,ed_weight[u]);

      DEBUG(cerr<<"added:\n";
            printEdge(edge));
    }

    //vt.erase(u);
    
    //remove u frm vt
    for(vector<Node *>::iterator VI=vt.begin(), VE=vt.end(); VI!=VE; ++VI){
      if(**VI==*u){
	vt.erase(VI);
	break;
      }
    }
    
    //assign wt(v) to all adjacent vertices v of u
    //only if v is in vt
    Graph::nodeList &nl = getNodeList(u);
    for(nodeList::iterator NI=nl.begin(), NE=nl.end(); NI!=NE; ++NI){
      Node *v=NI->element;
      int weight=-NI->weight;
      //check if v is in vt
      bool contains=false;
      for(vector<Node *>::iterator VI=vt.begin(), VE=vt.end(); VI!=VE; ++VI){
	if(**VI==*v){
	  contains=true;
	  break;
	}
      }
      DEBUG(cerr<<"wt:v->wt"<<weight<<":"<<v->getWeight()<<"\n";
            printNode(v);cerr<<"node wt:"<<(*v).weight<<"\n");

      //so if v in in vt, change wt(v) to wt(u->v)
      //only if wt(u->v)<wt(v)
      if(contains && weight<v->getWeight()){
	parent[v]=u;
	ed_weight[v]=weight;
	v->setWeight(weight);

	DEBUG(cerr<<v->getWeight()<<":Set weight------\n";
              printGraph();
              printEdge(Edge(u,v,weight)));
      }
    }
  }
  return st;
}

//print the graph (for debugging)   
void Graph::printGraph(){
   vector<Node *> lt=getAllNodes();
   cerr<<"Graph---------------------\n";
   for(vector<Node *>::iterator LI=lt.begin(), LE=lt.end(); LI!=LE; ++LI){
     cerr<<((*LI)->getElement())->getName()<<"->";
     Graph::nodeList &nl = getNodeList(*LI);
     for(Graph::nodeList::iterator NI=nl.begin(), NE=nl.end(); NI!=NE; ++NI){
       cerr<<":"<<"("<<(NI->element->getElement())
	 ->getName()<<":"<<NI->element->getWeight()<<","<<NI->weight<<")";
     }
     cerr<<"--------\n";
   }
}


//get a list of nodes in the graph
//in r-topological sorted order
//note that we assumed graph to be connected
vector<Node *> Graph::reverseTopologicalSort(){
  vector <Node *> toReturn;
  vector<Node *> lt=getAllNodes();
  for(vector<Node *>::iterator LI=lt.begin(), LE=lt.end(); LI!=LE; ++LI){
    if((*LI)->getWeight()!=GREY && (*LI)->getWeight()!=BLACK)
      DFS_Visit(*LI, toReturn);
  }

  return toReturn;
}

//a private method for doing DFS traversal of graph
//this is used in determining the reverse topological sort 
//of the graph
void Graph::DFS_Visit(Node *nd, vector<Node *> &toReturn){
  nd->setWeight(GREY);
  vector<Node *> lt=getSuccNodes(nd);
  for(vector<Node *>::iterator LI=lt.begin(), LE=lt.end(); LI!=LE; ++LI){
    if((*LI)->getWeight()!=GREY && (*LI)->getWeight()!=BLACK)
      DFS_Visit(*LI, toReturn);
  }
  toReturn.push_back(nd);
}

//Ordinarily, the graph is directional
//this converts the graph into an 
//undirectional graph
//This is done by adding an edge
//v->u for all existing edges u->v
void Graph::makeUnDirectional(){
  vector<Node* > allNodes=getAllNodes();
  for(vector<Node *>::iterator NI=allNodes.begin(), NE=allNodes.end(); NI!=NE; 
      ++NI) {
    nodeList &nl = getNodeList(*NI);
    for(nodeList::iterator NLI=nl.begin(), NLE=nl.end(); NLI!=NLE; ++NLI){
      Edge ed(NLI->element, *NI, NLI->weight);
      if(!hasEdgeAndWt(ed)){
	DEBUG(cerr<<"######doesn't hv\n";
              printEdge(ed));
	addEdgeForce(ed);
      }
    }
  }
}

//reverse the sign of weights on edges
//this way, max-spanning tree could be obtained
//using min-spanning tree, and vice versa
void Graph::reverseWts(){
  vector<Node *> allNodes=getAllNodes();
  for(vector<Node *>::iterator NI=allNodes.begin(), NE=allNodes.end(); NI!=NE; 
      ++NI) {
    nodeList &node_list = getNodeList(*NI);
    for(nodeList::iterator NLI=nodes[*NI].begin(), NLE=nodes[*NI].end(); 
	NLI!=NLE; ++NLI)
      NLI->weight=-NLI->weight;
  }
}


//getting the backedges in a graph
//Its a variation of DFS to get the backedges in the graph
//We get back edges by associating a time
//and a color with each vertex.
//The time of a vertex is the time when it was first visited
//The color of a vertex is initially WHITE,
//Changes to GREY when it is first visited,
//and changes to BLACK when ALL its neighbors
//have been visited
//So we have a back edge when we meet a successor of
//a node with smaller time, and GREY color
void Graph::getBackEdges(vector<Edge > &be, map<Node *, int> &d){
  map<Node *, Color > color;
  int time=0;

  getBackEdgesVisit(getRoot(), be, color, d, time);
}

//helper function to get back edges: it is called by 
//the "getBackEdges" function above
void Graph::getBackEdgesVisit(Node *u, vector<Edge > &be,
			      map<Node *, Color > &color,
			      map<Node *, int > &d, int &time) {
  color[u]=GREY;
  time++;
  d[u]=time;

  vector<graphListElement> &succ_list = getNodeList(u);
  
  for(vector<graphListElement>::iterator vl=succ_list.begin(), 
	ve=succ_list.end(); vl!=ve; ++vl){
    Node *v=vl->element;
    if(color[v]!=GREY && color[v]!=BLACK){
      getBackEdgesVisit(v, be, color, d, time);
    }
    
    //now checking for d and f vals
    if(color[v]==GREY){
      //so v is ancestor of u if time of u > time of v
      if(d[u] >= d[v]){
	Edge *ed=new Edge(u, v,vl->weight, vl->randId);
	if (!(*u == *getExit() && *v == *getRoot()))
	  be.push_back(*ed);      // choose the forward edges
      }
    }
  }
  color[u]=BLACK;//done with visiting the node and its neighbors
}


