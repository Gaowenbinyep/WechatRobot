from agent.agent_setup import agent, style_chain, router_chain

from graph_nodes.nodes import build_workflow




workflow = build_workflow()
    
result = workflow.invoke({})