import os
from dotenv import load_dotenv
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()
aws_region = os.getenv("AWS_REGION")
model_id = os.getenv("MODEL")

# Bedrock Claude setup
bedrock_llm = ChatBedrock(model=model_id, region=aws_region)

# =================== ACCOMMODATION TOOLS ===================
@tool
def find_hotels(destination: str, stars: int, budget: str, dates: str) -> str:
    """Find hotels based on location, rating, budget and dates."""
    return f"Top {stars}-star hotels in {destination} for {dates} within {budget} budget: The Plaza Hotel ($450/night), The St. Regis ($380/night), The Ritz-Carlton ($520/night)."

@tool
def book_accommodation(hotel_name: str, dates: str, room_type: str) -> str:
    """Book hotel accommodation."""
    return f"Booking confirmed at {hotel_name} for {dates}. {room_type} reserved. Confirmation #HTL12345."

# =================== ACTIVITIES TOOLS ===================
@tool
def find_activities(destination: str, interests: List[str], dates: str) -> str:
    """Find activities based on interests and travel dates."""
    activity_map = {
        "music": ["Lincoln Center concerts", "Blue Note Jazz Club"],
        "art": ["Metropolitan Museum", "MoMA"],
        "nightlife": ["Rooftop bars in Manhattan", "Speakeasies in East Village"]
    }
    
    activities = []
    for interest in interests:
        if interest.lower() in activity_map:
            activities.extend(activity_map[interest.lower()])
    
    return f"Recommended activities in {destination}: {', '.join(activities)}"

@tool
def book_activity_tickets(activity: str, date: str, quantity: int) -> str:
    """Book tickets for activities and events."""
    return f"Tickets booked for {activity} on {date}. Quantity: {quantity}. Confirmation #ACT67890."

# =================== ACCOMMODATION SUBGRAPH ===================
def accommodation_agent(state):
    messages = state["messages"]
    human_messages = [msg for msg in messages if msg.type == "human"]
    system_msg = SystemMessage(content="You are a hotel booking specialist. Help find and book accommodations only. Provide names and details of hotels only. No Shedules or activities required.")
    
    llm_with_tools = bedrock_llm.bind_tools([find_hotels, book_accommodation])
    response = llm_with_tools.invoke([system_msg] + human_messages)
    
    # If tools were called, execute them and get final response
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_node = ToolNode([find_hotels, book_accommodation])
        tool_result = tool_node.invoke({"messages": [system_msg] + human_messages + [response]})
        
        # Get final response after tools - keep tools bound
        final_response = llm_with_tools.invoke([system_msg] + human_messages + [response] + tool_result["messages"])
        return {"messages": [response] + tool_result["messages"] + [final_response]}
    
    return {"messages": [response]}

# Create simple accommodation subgraph
accommodation_subgraph = StateGraph(MessagesState)
accommodation_subgraph.add_node("agent", accommodation_agent)
accommodation_subgraph.add_edge(START, "agent")
accommodation_subgraph.add_edge("agent", END)

accommodation_graph = accommodation_subgraph.compile()

# =================== ACTIVITIES SUBGRAPH ===================
def activity_agent(state):
    messages = state["messages"]
    human_messages = [msg for msg in messages if msg.type == "human"]
    system_msg = SystemMessage(content="You are an activities specialist. Help find and book entertainment and activities only.")
    
    llm_with_tools = bedrock_llm.bind_tools([find_activities, book_activity_tickets])
    response = llm_with_tools.invoke([system_msg] + human_messages)
    
    # If tools were called, execute them and get final response
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_node = ToolNode([find_activities, book_activity_tickets])
        tool_result = tool_node.invoke({"messages": [system_msg] + human_messages + [response]})
        
        # Get final response after tools - keep tools bound
        final_response = llm_with_tools.invoke([system_msg] + human_messages + [response] + tool_result["messages"])
        return {"messages": [response] + tool_result["messages"] + [final_response]}
    
    return {"messages": [response]}

# Create simple activities subgraph
activity_subgraph = StateGraph(MessagesState)
activity_subgraph.add_node("agent", activity_agent)
activity_subgraph.add_edge(START, "agent")
activity_subgraph.add_edge("agent", END)

activity_graph = activity_subgraph.compile()

# =================== MAIN TRAVEL PLANNER ===================
def accommodation_node(state):
    """Run accommodation subgraph"""
    # Pass only the original human message to avoid assistant message conflicts
    human_messages = [msg for msg in state["messages"] if msg.type == "human"]
    result = accommodation_graph.invoke({"messages": human_messages})
    return {"messages": state["messages"] + result["messages"]}

def activity_node(state):
    """Run activities subgraph"""
    # Pass only the original human message to avoid assistant message conflicts
    human_messages = [msg for msg in state["messages"] if msg.type == "human"]
    result = activity_graph.invoke({"messages": human_messages})
    return {"messages": state["messages"] + result["messages"]}

def coordinator_node(state):
    """Create final travel plan"""
    messages = state["messages"]
    
    # Get original request
    original_request = messages[0].content if messages else ""
    
    # Collect all agent responses
    agent_responses = [msg.content for msg in messages[1:] if hasattr(msg, 'content') and msg.content]
    
    summary_request = f"""
    Create a comprehensive travel plan based on:
    
    Original Request: {original_request}
    
    Recommendations: {' '.join(agent_responses)}
    
    Organize into: Accommodation, Activities, Summary
    """
    
    response = bedrock_llm.invoke([HumanMessage(content=summary_request)])
    return {"messages": state["messages"] + [response]}

# Build main travel planner graph
travel_planner_graph = StateGraph(MessagesState)

# Add nodes
travel_planner_graph.add_node("accommodation", accommodation_node)
travel_planner_graph.add_node("activities", activity_node)
travel_planner_graph.add_node("coordinator", coordinator_node)

# Add edges
travel_planner_graph.add_edge(START, "accommodation")
travel_planner_graph.add_edge("accommodation", "activities")
travel_planner_graph.add_edge("activities", "coordinator")
travel_planner_graph.add_edge("coordinator", END)

# Compile main graph
travel_planner = travel_planner_graph.compile()

# =================== EXECUTION ===================
if __name__ == "__main__":
    user_request = HumanMessage(content="""
    I'm planning a 3-day trip to New York City from December 25-27th. 
    I love music, art, and nightlife. 
    Looking for 4-5 star hotel recommendations and complete travel plan.
    """)
    
    # print("🚀 Travel Planner:")
    # result = travel_planner.invoke({"messages": [user_request]})
    
    # print("\n📋 Final Travel Plan:")
    # print(result['messages'][-1].content)

    result = accommodation_graph.invoke({"messages": [user_request]})
    print("\n🏨 Accommodation Subgraph Result:"
          f"\n{result['messages'][-1].content}")
    
    result = travel_planner.nodes['accommodation'].invoke({"messages": [user_request]})
    print("\n🏨 Accommodation Node Result:"
          f"\n{result['messages'][-1].content}")
    

    # streams

    # """Stream the travel planner execution with real-time updates"""
    # user_request = HumanMessage(content="""
    # I'm planning a 3-day trip to New York City from December 25-27th. 
    # I love music, art, and nightlife. 
    # Looking for 4-5 star hotel recommendations and complete travel plan.
    # """)
    
    print("Starting Travel Planner (Streaming Mode)...")
    print("=" * 50)
    
    # Stream the graph execution
    for event in travel_planner.stream({"messages": [user_request]}):
        for node_name, node_result in event.items():
            print(f"\nNode: {node_name.upper()}")
            print("-" * 30)
            
            # Get the latest message from this node
            if 'messages' in node_result and node_result['messages']:
                latest_message = node_result['messages'][-1]
                if hasattr(latest_message, 'content') and latest_message.content:
                    print(latest_message.content)
            
            print("-" * 30)
    
    print("\n✅ Travel Planner streaming completed!")
    


    # Streaming individual components
    """Simple streaming example for individual components"""
    user_request = HumanMessage(content="""
    I need a 4-star hotel in New York for December 25-27th.
    """)
    
    print("\nStreaming Accommodation Search")
    print("=" * 40)
    
    # Stream accommodation subgraph
    for event in accommodation_graph.stream({"messages": [user_request]}):
        for node_name, node_result in event.items():
            if 'messages' in node_result and node_result['messages']:
                latest_message = node_result['messages'][-1]
                if hasattr(latest_message, 'content'):
                    print(f"{node_name}: {latest_message.content}...")