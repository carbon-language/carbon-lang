//===-- GrapAuxillary.cpp- Auxillary functions on graph ----------*- C++ -*--=//
//
//auxillary function associated with graph: they
//all operate on graph, and help in inserting
//instrumentation for trace generation
//
//===----------------------------------------------------------------------===//

#include "Graph.h"
#include "llvm/BasicBlock.h"
#include <algorithm>
#include <iostream>

using std::list;
using std::map;
using std::vector;
using std::cerr;

//check if 2 edges are equal (same endpoints and same weight)
static bool edgesEqual(Edge  ed1, Edge ed2){
  return ((ed1==ed2) && ed1.getWeight()==ed2.getWeight());
}

//Get the vector of edges that are to be instrumented in the graph
static void getChords(vector<Edge > &chords, Graph &g, Graph st){
  //make sure the spanning tree is directional
  //iterate over ALL the edges of the graph
  list<Node *> allNodes=g.getAllNodes();
  for(list<Node *>::iterator NI=allNodes.begin(), NE=allNodes.end(); NI!=NE; 
      ++NI){
    Graph::nodeList node_list=g.getNodeList(*NI);
    for(Graph::nodeList::iterator NLI=node_list.begin(), NLE=node_list.end(); 
	NLI!=NLE; ++NLI){
      Edge f(*NI, NLI->element,NLI->weight);
      if(!(st.hasEdgeAndWt(f)))//addnl
	chords.push_back(f);
    }
  }
}

//Given a tree t, and a "directed graph" g
//replace the edges in the tree t with edges that exist in graph
//The tree is formed from "undirectional" copy of graph
//So whatever edges the tree has, the undirectional graph 
//would have too. This function corrects some of the directions in 
//the tree so that now, all edge directions in the tree match
//the edge directions of corresponding edges in the directed graph
static void removeTreeEdges(Graph &g, Graph& t){
  list<Node* > allNodes=t.getAllNodes();
  for(list<Node *>::iterator NI=allNodes.begin(), NE=allNodes.end(); NI!=NE; 
      ++NI){
    Graph::nodeList nl=t.getNodeList(*NI);
    for(Graph::nodeList::iterator NLI=nl.begin(), NLE=nl.end();	NLI!=NLE;++NLI){
      Edge ed(NLI->element, *NI, NLI->weight);
      //if(!g.hasEdge(ed)) t.removeEdge(ed);
      if(!g.hasEdgeAndWt(ed)) t.removeEdge(ed);//tree has only one edge
      //between any pair of vertices, so no need to delete by edge wt
    }
  }
}

//Assign a value to all the edges in the graph
//such that if we traverse along any path from root to exit, and
//add up the edge values, we get a path number that uniquely
//refers to the path we travelled
int valueAssignmentToEdges(Graph& g){
  list<Node *> revtop=g.reverseTopologicalSort();
  map<Node *,int > NumPaths;
  for(list<Node *>::iterator RI=revtop.begin(), RE=revtop.end(); RI!=RE; ++RI){
    if(g.isLeaf(*RI))
      NumPaths[*RI]=1;
    else{
      NumPaths[*RI]=0;
      list<Node *> succ=g.getSuccNodes(*RI);
      for(list<Node *>::iterator SI=succ.begin(), SE=succ.end(); SI!=SE; ++SI){
	Edge ed(*RI,*SI,NumPaths[*RI]);
	g.setWeight(ed);
	NumPaths[*RI]+=NumPaths[*SI];
      }
    }
  }
  return NumPaths[g.getRoot()];
}

//This is a helper function to get the edge increments
//This is used in conjuntion with inc_DFS
//to get the edge increments
//Edge increment implies assigning a value to all the edges in the graph
//such that if we traverse along any path from root to exit, and
//add up the edge values, we get a path number that uniquely
//refers to the path we travelled
//inc_Dir tells whether 2 edges are in same, or in different directions
//if same direction, return 1, else -1
static int inc_Dir(Edge e, Edge f){ 
 if(e.isNull()) 
    return 1;
 
 //check that the edges must have atleast one common endpoint
  assert(*(e.getFirst())==*(f.getFirst()) ||
	 *(e.getFirst())==*(f.getSecond()) || 
	 *(e.getSecond())==*(f.getFirst()) ||
	 *(e.getSecond())==*(f.getSecond()));

  if(*(e.getFirst())==*(f.getSecond()) || 
     *(e.getSecond())==*(f.getFirst()))
    return 1;
  
  return -1;
}

//used for getting edge increments (read comments above in inc_Dir)
//inc_DFS is a modification of DFS 
static void inc_DFS(Graph& g,Graph& t,map<Edge, int>& Increment, 
	     int events, Node *v, Edge e){
  
  list<Node *> allNodes=t.getAllNodes();
  
  for(list<Node *>::iterator NI=allNodes.begin(), NE=allNodes.end(); NI!=NE; 
      ++NI){
    Graph::nodeList node_list=t.getNodeList(*NI);
    for(Graph::nodeList::iterator NLI=node_list.begin(), NLE=node_list.end(); 
	NLI!= NLE; ++NLI){
      Edge f(*NI, NLI->element,NLI->weight);
      if(!edgesEqual(f,e) && *v==*(f.getSecond())){
	int dir_count=inc_Dir(e,f);
	int wt=1*f.getWeight();
	inc_DFS(g,t, Increment, dir_count*events+wt, f.getFirst(), f);
      }
    }
  }

  for(list<Node *>::iterator NI=allNodes.begin(), NE=allNodes.end(); NI!=NE; 
      ++NI){
    Graph::nodeList node_list=t.getNodeList(*NI);
    for(Graph::nodeList::iterator NLI=node_list.begin(), NLE=node_list.end(); 
	NLI!=NLE; ++NLI){
      Edge f(*NI, NLI->element,NLI->weight);
      if(!edgesEqual(f,e) && *v==*(f.getFirst())){
      	int dir_count=inc_Dir(e,f);
	int wt=1*f.getWeight();
	inc_DFS(g,t, Increment, dir_count*events+wt, 
		f.getSecond(), f);
      }
    }
  }

  allNodes=g.getAllNodes();
  for(list<Node *>::iterator NI=allNodes.begin(), NE=allNodes.end(); NI!=NE; 
      ++NI){
    Graph::nodeList node_list=g.getNodeList(*NI);
    for(Graph::nodeList::iterator NLI=node_list.begin(), NLE=node_list.end(); 
	NLI!=NLE; ++NLI){
      Edge f(*NI, NLI->element,NLI->weight);
      if(!(t.hasEdgeAndWt(f)) && (*v==*(f.getSecond()) || 
				  *v==*(f.getFirst()))){
	int dir_count=inc_Dir(e,f);
	Increment[f]+=dir_count*events;
      }
    }
  }
}

//Now we select a subset of all edges
//and assign them some values such that 
//if we consider just this subset, it still represents
//the path sum along any path in the graph
static map<Edge, int> getEdgeIncrements(Graph& g, Graph& t){
  //get all edges in g-t
  map<Edge, int> Increment;

  list<Node *> allNodes=g.getAllNodes();
 
  for(list<Node *>::iterator NI=allNodes.begin(), NE=allNodes.end(); NI!=NE; 
      ++NI){
    Graph::nodeList node_list=g.getNodeList(*NI);
    for(Graph::nodeList::iterator NLI=node_list.begin(), NLE=node_list.end(); 
	NLI!=NLE; ++NLI){
      Edge ed(*NI, NLI->element,NLI->weight);
      if(!(t.hasEdge(ed))){
	Increment[ed]=0;;
      }
    }
  }

  Edge *ed=new Edge();
  inc_DFS(g,t,Increment, 0, g.getRoot(), *ed);


  for(list<Node *>::iterator NI=allNodes.begin(), NE=allNodes.end(); NI!=NE; 
      ++NI){
    Graph::nodeList node_list=g.getNodeList(*NI);
    for(Graph::nodeList::iterator NLI=node_list.begin(), NLE=node_list.end(); 
	NLI!=NLE; ++NLI){
      Edge ed(*NI, NLI->element,NLI->weight);
      if(!(t.hasEdge(ed))){
	int wt=ed.getWeight();
	Increment[ed]+=wt;
      }
    }
  }

  return Increment;
}

//Based on edgeIncrements (above), now obtain
//the kind of code to be inserted along an edge
//The idea here is to minimize the computation
//by inserting only the needed code
static void getCodeInsertions(Graph &g, map<Edge, getEdgeCode *> &instr,
                              vector<Edge > &chords, 
                              map<Edge,int> &edIncrements){

  //Register initialization code
  vector<Node *> ws;
  ws.push_back(g.getRoot());
  while(ws.size()>0){
    Node *v=ws.back();
    ws.pop_back();
    //for each edge v->w
    Graph::nodeList succs=g.getNodeList(v);
    
    for(Graph::nodeList::iterator nl=succs.begin(), ne=succs.end();
	nl!=ne; ++nl){
      int edgeWt=nl->weight;
      Node *w=nl->element;
      //if chords has v->w
      Edge ed(v,w);
      
      bool hasEdge=false;
      for(vector<Edge>::iterator CI=chords.begin(), CE=chords.end();
	  CI!=CE && !hasEdge;++CI){
	if(*CI==ed){
	  hasEdge=true;
	}
      }
      if(hasEdge){
	getEdgeCode *edCd=new getEdgeCode();
	edCd->setCond(1);
	edCd->setInc(edIncrements[ed]);
	instr[ed]=edCd;
      }
      else if((g.getPredNodes(w)).size()==1){
	ws.push_back(w);
      }
      else{
	getEdgeCode *edCd=new getEdgeCode();
	edCd->setCond(2);
	edCd->setInc(0);
	instr[ed]=edCd;
      }
    }
  }

  /////Memory increment code
  ws.push_back(g.getExit());
  
  while(!ws.empty()) {
    Node *w=ws.back();
    ws.pop_back();
    
    //for each edge v->w
    list<Node *> preds=g.getPredNodes(w);
    for(list<Node *>::iterator pd=preds.begin(), pe=preds.end(); pd!=pe; ++pd){
      Node *v=*pd;
      //if chords has v->w
    
      Edge ed(v,w);
      getEdgeCode *edCd=new getEdgeCode();
      bool hasEdge=false;
      for(vector<Edge>::iterator CI=chords.begin(), CE=chords.end(); CI!=CE;
	  ++CI){
	if(*CI==ed){
	  hasEdge=true;
	  break;
	}
      }
      if(hasEdge){
	char str[100];
	if(instr[ed]!=NULL && instr[ed]->getCond()==1){
	  instr[ed]->setCond(4);
	}
	else{
	  edCd->setCond(5);
	  edCd->setInc(edIncrements[ed]);
	  instr[ed]=edCd;
	}
	
      }
      else if(g.getSuccNodes(v).size()==1)
	ws.push_back(v);
      else{
	edCd->setCond(6);
	instr[ed]=edCd;
      }
    }
  }

  ///// Register increment code
  for(vector<Edge>::iterator CI=chords.begin(), CE=chords.end(); CI!=CE; ++CI){
    getEdgeCode *edCd=new getEdgeCode();
    if(instr[*CI]==NULL){
      edCd->setCond(3);
      edCd->setInc(edIncrements[*CI]);
      instr[*CI]=edCd;
    }
  }
}

//Add dummy edges corresponding to the back edges
//If a->b is a backedge
//then incoming dummy edge is root->b
//and outgoing dummy edge is a->exit
void addDummyEdges(vector<Edge > &stDummy, 
		   vector<Edge > &exDummy, 
		   Graph &g, vector<Edge> &be){
  for(vector<Edge >::iterator VI=be.begin(), VE=be.end(); VI!=VE; ++VI){
    Edge ed=*VI;
    Node *first=ed.getFirst();
    Node *second=ed.getSecond();
    g.removeEdge(ed);

    if(!(*second==*(g.getRoot()))){
      Edge *st=new Edge(g.getRoot(), second); 
      
      //check if stDummy doesn't have it already
      if(find(stDummy.begin(), stDummy.end(), *st) == stDummy.end())
	stDummy.push_back(*st);
      g.addEdgeForce(*st);
    }

    if(!(*first==*(g.getExit()))){
      Edge *ex=new Edge(first, g.getExit());
      
      if (find(exDummy.begin(), exDummy.end(), *ex) == exDummy.end()) {
	exDummy.push_back(*ex);
	g.addEdgeForce(*ex);
      }
    }
  }
}

//print a given edge in the form BB1Label->BB2Label
void printEdge(Edge ed){
  cerr<<((ed.getFirst())->getElement())
    ->getName()<<"->"<<((ed.getSecond())
			  ->getElement())->getName()<<
    ":"<<ed.getWeight()<<"\n";
}

//Move the incoming dummy edge code and outgoing dummy
//edge code over to the corresponding back edge
static void moveDummyCode(const vector<Edge> &stDummy, 
                          const vector<Edge> &exDummy, 
                          const vector<Edge> &be,  
                          map<Edge, getEdgeCode *> &insertions){
  typedef vector<Edge >::const_iterator vec_iter;
  
  DEBUG( //print all back, st and ex dummy
        cerr<<"BackEdges---------------\n";
        for(vec_iter VI=be.begin(); VI!=be.end(); ++VI)
        printEdge(*VI);
        cerr<<"StEdges---------------\n";
        for(vec_iter VI=stDummy.begin(); VI!=stDummy.end(); ++VI)
        printEdge(*VI);
        cerr<<"ExitEdges---------------\n";
        for(vec_iter VI=exDummy.begin(); VI!=exDummy.end(); ++VI)
        printEdge(*VI);
        cerr<<"------end all edges\n");

  std::vector<Edge> toErase;
  for(map<Edge,getEdgeCode *>::iterator MI=insertions.begin(), 
	ME=insertions.end(); MI!=ME; ++MI){
    Edge ed=MI->first;
    getEdgeCode *edCd=MI->second;
    bool dummyHasIt=false;

    DEBUG(cerr<<"Current edge considered---\n";
          printEdge(ed));

    //now check if stDummy has ed
    for(vec_iter VI=stDummy.begin(), VE=stDummy.end(); VI!=VE && !dummyHasIt; 
	++VI){
      if(*VI==ed){
	DEBUG(cerr<<"Edge matched with stDummy\n");

	dummyHasIt=true;
	bool dummyInBe=false;
	//dummy edge with code
	for(vec_iter BE=be.begin(), BEE=be.end(); BE!=BEE && !dummyInBe; ++BE){
	  Edge backEdge=*BE;
	  Node *st=backEdge.getSecond();
	  Node *dm=ed.getSecond();
	  if(*dm==*st){
	    //so this is the back edge to use
	    DEBUG(cerr<<"Moving to backedge\n";
                  printEdge(backEdge));

	    getEdgeCode *ged=new getEdgeCode();
	    ged->setCdIn(edCd);
	    toErase.push_back(ed);
	    insertions[backEdge]=ged;
	    dummyInBe=true;
	  }
	}
	assert(dummyInBe);
      }
    }
    if(!dummyHasIt){
      //so exDummy may hv it
      bool inExDummy=false;
      for(vec_iter VI=exDummy.begin(), VE=exDummy.end(); VI!=VE && !inExDummy; 
	  ++VI){
	if(*VI==ed){
	  inExDummy=true;
	  DEBUG(cerr<<"Edge matched with exDummy\n");
	  bool dummyInBe2=false;
	  //dummy edge with code
	  for(vec_iter BE=be.begin(), BEE=be.end(); BE!=BEE && !dummyInBe2; 
	      ++BE){
	    Edge backEdge=*BE;
	    Node *st=backEdge.getFirst();
	    Node *dm=ed.getFirst();
	    if(*dm==*st){
	      //so this is the back edge to use
	      getEdgeCode *ged;
	      if(insertions[backEdge]==NULL)
		ged=new getEdgeCode();
	      else
		ged=insertions[backEdge];
	      toErase.push_back(ed);
	      ged->setCdOut(edCd);
	      insertions[backEdge]=ged;
	      dummyInBe2=true;
	    }
	  }
	  assert(dummyInBe2);
	}
      }
    }
  }

  DEBUG(cerr<<"size of deletions: "<<toErase.size()<<"\n");

  for(vector<Edge >::iterator vmi=toErase.begin(), vme=toErase.end(); vmi!=vme; 
      ++vmi)
    insertions.erase(*vmi);

  DEBUG(cerr<<"SIZE OF INSERTIONS AFTER DEL "<<insertions.size()<<"\n");
}

//Do graph processing: to determine minimal edge increments, 
//appropriate code insertions etc and insert the code at
//appropriate locations
void processGraph(Graph &g, 
		  Instruction *rInst, 
		  Instruction *countInst, 
		  vector<Edge >& be, 
		  vector<Edge >& stDummy, 
		  vector<Edge >& exDummy){
  //Given a graph: with exit->root edge, do the following in seq:
  //1. get back edges
  //2. insert dummy edges and remove back edges
  //3. get edge assignments
  //4. Get Max spanning tree of graph:
  //   -Make graph g2=g undirectional
  //   -Get Max spanning tree t
  //   -Make t undirectional
  //   -remove edges from t not in graph g
  //5. Get edge increments
  //6. Get code insertions
  //7. move code on dummy edges over to the back edges
  

  //This is used as maximum "weight" for 
  //priority queue
  //This would hold all 
  //right as long as number of paths in the graph
  //is less than this
  const int INFINITY=99999999;


  //step 1-3 are already done on the graph when this function is called
  DEBUG(printGraph(g));

  //step 4: Get Max spanning tree of graph

  //now insert exit to root edge
  //if its there earlier, remove it!
  //assign it weight INFINITY
  //so that this edge IS ALWAYS IN spanning tree
  //Note than edges in spanning tree do not get 
  //instrumented: and we do not want the
  //edge exit->root to get instrumented
  //as it MAY BE a dummy edge
  Edge ed(g.getExit(),g.getRoot(),INFINITY);
  g.addEdge(ed,INFINITY);
  Graph g2=g;

  //make g2 undirectional: this gives a better
  //maximal spanning tree
  g2.makeUnDirectional();
  DEBUG(printGraph(g2));

  Graph *t=g2.getMaxSpanningTree();
  DEBUG(printGraph(*t));

  //now edges of tree t have weights reversed
  //(negative) because the algorithm used
  //to find max spanning tree is 
  //actually for finding min spanning tree
  //so get back the original weights
  t->reverseWts();

  //Ordinarily, the graph is directional
  //lets converts the graph into an 
  //undirectional graph
  //This is done by adding an edge
  //v->u for all existing edges u->v
  t->makeUnDirectional();

  //Given a tree t, and a "directed graph" g
  //replace the edges in the tree t with edges that exist in graph
  //The tree is formed from "undirectional" copy of graph
  //So whatever edges the tree has, the undirectional graph 
  //would have too. This function corrects some of the directions in 
  //the tree so that now, all edge directions in the tree match
  //the edge directions of corresponding edges in the directed graph
  removeTreeEdges(g, *t);

  DEBUG(cerr<<"Spanning tree---------\n";
        printGraph(*t);
        cerr<<"-------end spanning tree\n");

  //now remove the exit->root node
  //and re-add it with weight 0
  //since infinite weight is kinda confusing
  g.removeEdge(ed);
  Edge edNew(g.getExit(), g.getRoot(),0);
  g.addEdge(edNew,0);
  if(t->hasEdge(ed)){
    t->removeEdge(ed);
    t->addEdge(edNew,0);
  }

  DEBUG(printGraph(g);
        printGraph(*t));

  //step 5: Get edge increments

  //Now we select a subset of all edges
  //and assign them some values such that 
  //if we consider just this subset, it still represents
  //the path sum along any path in the graph
  map<Edge, int> increment=getEdgeIncrements(g,*t);

  DEBUG(//print edge increments for debugging
        for(map<Edge, int>::iterator MI=increment.begin(), ME = increment.end();
            MI != ME; ++MI) {
          printEdge(MI->first);
          cerr << "Increment for above:" << MI->second << "\n";
        });
 
  //step 6: Get code insertions
  
  //Based on edgeIncrements (above), now obtain
  //the kind of code to be inserted along an edge
  //The idea here is to minimize the computation
  //by inserting only the needed code
  vector<Edge> chords;
  getChords(chords, g, *t);

  map<Edge, getEdgeCode *> codeInsertions;
  getCodeInsertions(g, codeInsertions, chords,increment);
  
  DEBUG (//print edges with code for debugging
         cerr<<"Code inserted in following---------------\n";
         for(map<Edge, getEdgeCode *>::iterator cd_i=codeInsertions.begin(), 
               cd_e=codeInsertions.end(); cd_i!=cd_e; ++cd_i){
           printEdge(cd_i->first);
           cerr<<cd_i->second->getCond()<<":"<<cd_i->second->getInc()<<"\n";
         }
         cerr<<"-----end insertions\n");

  //step 7: move code on dummy edges over to the back edges

  //Move the incoming dummy edge code and outgoing dummy
  //edge code over to the corresponding back edge
  moveDummyCode(stDummy, exDummy, be, codeInsertions);
  
  DEBUG(//debugging info
        cerr<<"After moving dummy code\n";
        for(map<Edge, getEdgeCode *>::iterator cd_i=codeInsertions.begin(), 
              cd_e=codeInsertions.end(); cd_i != cd_e; ++cd_i){
          printEdge(cd_i->first);
          cerr<<cd_i->second->getCond()<<":"
              <<cd_i->second->getInc()<<"\n";
        }
        cerr<<"Dummy end------------\n");

  //see what it looks like...
  //now insert code along edges which have codes on them
  for(map<Edge, getEdgeCode *>::iterator MI=codeInsertions.begin(), 
	ME=codeInsertions.end(); MI!=ME; ++MI){
    Edge ed=MI->first;
    insertBB(ed, MI->second, rInst, countInst);
  } 
}



//print the graph (for debugging)
void printGraph(Graph &g){
  list<Node *> lt=g.getAllNodes();
  cerr<<"Graph---------------------\n";
  for(list<Node *>::iterator LI=lt.begin(); 
      LI!=lt.end(); ++LI){
    cerr<<((*LI)->getElement())->getName()<<"->";
    Graph::nodeList nl=g.getNodeList(*LI);
    for(Graph::nodeList::iterator NI=nl.begin(); 
	NI!=nl.end(); ++NI){
      cerr<<":"<<"("<<(NI->element->getElement())
	->getName()<<":"<<NI->element->getWeight()<<","<<NI->weight<<")";
    }
    cerr<<"\n";
  }
  cerr<<"--------------------Graph\n";
}
