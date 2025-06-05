import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langsmith import Client
from langsmith.evaluation import evaluate

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

# =================== EVALUATION SETUP ===================
def create_evaluation_dataset():
    """Create evaluation dataset in LangSmith"""
    ls_client = Client()
    dataset_name = "travel_planner_eval"
    
    try:
        # Check if dataset exists
        datasets = list(ls_client.list_datasets(dataset_name=dataset_name))
        dataset = next((d for d in datasets if d.name == dataset_name), None)
        
        if dataset is None:
            print(f"Creating dataset: {dataset_name}")
            dataset = ls_client.create_dataset(
                dataset_name=dataset_name,
                description="Travel planner evaluation dataset"
            )
            
            # Add test cases
            test_cases = [
                {
                    "inputs": {"question": "I need a 4-star hotel in New York for December 25-27th with activities for music lovers."},
                    "outputs": {"answer": "A comprehensive travel plan including 4-star hotel recommendations in New York for December 25-27th and music-related activities such as concerts and jazz clubs."}
                },
                {
                    "inputs": {"question": "Find a 5-star hotel in Paris for January 10-12th with art activities."},
                    "outputs": {"answer": "A comprehensive travel plan including 5-star hotel recommendations in Paris for January 10-12th and art-related activities such as museum visits."}
                },
                {
                    "inputs": {"question": "Book a 3-star hotel in Tokyo for February 1-3rd with nightlife activities."},
                    "outputs": {"answer": "A comprehensive travel plan including 3-star Tokyo hotels and nightlife activities such as bars and clubs."}
                }
            ]
            
            for case in test_cases:
                ls_client.create_example(
                    inputs=case["inputs"],
                    outputs=case["outputs"],
                    dataset_id=dataset.id
                )
            
            print(f"Created dataset with {len(test_cases)} examples")
        else:
            print(f"Using existing dataset: {dataset_name}")
            
        return dataset
    
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        raise

# 3. Define the interface to your app
def travel_chatbot(inputs: dict) -> dict:
    """Interface function for travel planner evaluation"""
    # Convert input to graph state
    state = {"messages": [HumanMessage(content=inputs['question'])]}
    
    # Run travel planner
    result = travel_planner.invoke(state)
    
    # Extract the final answer
    final_answer = result["messages"][-1].content
    
    # Return in expected format
    return {"answer": final_answer}

# =================== EVALUATORS ===================
# 2. Define evaluators
def has_hotel_recommendations(outputs: dict, reference_outputs: dict) -> bool:
    """Check if response contains hotel recommendations"""
    answer = outputs.get("answer", "")
    hotel_keywords = ['hotel', 'accommodation', 'plaza', 'ritz', 'regis', 'booking', 'room']
    return any(keyword in answer.lower() for keyword in hotel_keywords)

def has_activity_suggestions(outputs: dict, reference_outputs: dict) -> bool:
    """Check if response contains activity suggestions"""
    answer = outputs.get("answer", "")
    activity_keywords = ['activity', 'activities', 'museum', 'concert', 'club', 'entertainment', 'attraction']
    return any(keyword in answer.lower() for keyword in activity_keywords)

def is_comprehensive_plan(outputs: dict, reference_outputs: dict) -> bool:
    """Check if the plan is comprehensive (has both hotels and activities)"""
    return (has_hotel_recommendations(outputs, reference_outputs) and 
            has_activity_suggestions(outputs, reference_outputs))

# =================== CSV EXPORT FUNCTION ===================
def export_to_csv(experiment, filename=None):
    """Simple CSV export of evaluation results"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"travel_planner_results_{timestamp}.csv"
    
    try:
        # Convert experiment to pandas DataFrame
        df = experiment.to_pandas()
        df.to_csv(filename, index=False)
        print(f"✅ Results exported to: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Pandas export failed: {e}")
        
        # Simple fallback - export basic results
        try:
            results = list(experiment)
            data = []
            
            for i, result in enumerate(results):
                row = {'test_id': i + 1}
                
                # Get input question
                try:
                    row['question'] = result['example'].inputs['question']
                except:
                    row['question'] = 'N/A'
                
                # Get output answer
                try:
                    row['answer'] = result['run'].outputs['answer']
                except:
                    row['answer'] = 'N/A'
                
                # Get evaluation scores
                try:
                    eval_results = result['evaluation_results']['results']
                    for eval_result in eval_results:
                        row[eval_result.key] = eval_result.score
                except:
                    row['has_hotel_recommendations'] = False
                    row['has_activity_suggestions'] = False
                    row['is_comprehensive_plan'] = False
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"✅ Fallback export completed: {filename}")
            return filename
            
        except Exception as e2:
            print(f"❌ All export methods failed: {e2}")
            return None

# =================== RUN EVALUATION ===================
def run_langsmith_evaluation():
    """Run the LangSmith evaluation following the reference pattern"""
    print("Setting up LangSmith evaluation...")
    
    # 1. Create and/or select your dataset
    ls_client = Client()
    dataset = create_evaluation_dataset()
    
    print("Running evaluation with LangSmith...")
    
    # 4. Run an evaluation
    experiment = ls_client.evaluate(
        travel_chatbot,  # Our travel planner interface
        data=dataset,
        evaluators=[
            has_hotel_recommendations,
            has_activity_suggestions,
            is_comprehensive_plan
        ],
        experiment_prefix="travel-planner-eval",
        upload_results=True,  # Set to True if you want to upload to LangSmith
        max_concurrency=1  # Keep low for rate limits
    )
    
    # 5. Analyze results locally
    print("\nAnalyzing results...")
    results = list(experiment)
    
    # Print summary statistics
    total_tests = len(results)
    print(f"\nTotal test cases: {total_tests}")
    
    # Check individual evaluator results
    hotel_passed = 0
    activity_passed = 0
    comprehensive_passed = 0
    
    for result in results:
        eval_results = result["evaluation_results"]["results"]
        
        for eval_result in eval_results:
            if eval_result.key == "has_hotel_recommendations" and eval_result.score:
                hotel_passed += 1
            elif eval_result.key == "has_activity_suggestions" and eval_result.score:
                activity_passed += 1
            elif eval_result.key == "is_comprehensive_plan" and eval_result.score:
                comprehensive_passed += 1
    
    print(f"Hotel recommendations: {hotel_passed}/{total_tests} ({hotel_passed/total_tests*100:.1f}%)")
    print(f"Activity suggestions: {activity_passed}/{total_tests} ({activity_passed/total_tests*100:.1f}%)")
    print(f"Comprehensive plans: {comprehensive_passed}/{total_tests} ({comprehensive_passed/total_tests*100:.1f}%)")
    
    # Check for failed comprehensive plans
    failed_comprehensive = [r for r in results 
                          if not any(eval_r.score for eval_r in r["evaluation_results"]["results"] 
                                   if eval_r.key == "is_comprehensive_plan")]
    
    if failed_comprehensive:
        print(f"\n❌ Failed comprehensive plans ({len(failed_comprehensive)}):")
        for i, r in enumerate(failed_comprehensive):
            print(f"\nFailed Test {i+1}:")
            print(f"Input: {r['example'].inputs['question']}")
            print(f"Output: {r['run'].outputs['answer'][:200]}...")
    
    # Export to CSV
    print("\n📊 Exporting results to CSV...")
    csv_file = export_to_csv(experiment)
    
    print("\n✅ LangSmith evaluation completed!")
    return experiment

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    print("Travel Planner LangSmith Evaluation with CSV Export")
    print("="*60)
    
    # Test basic functionality
    print("Testing basic functionality...")
    test_input = {"question": "I need a 4-star hotel in New York with music activities."}
    
    try:
        result = travel_chatbot(test_input)
        print("✅ Basic test passed")
        print(f"Sample output: {result['answer']}")
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        exit(1)
    
    # Run LangSmith evaluation
    try:
        results = run_langsmith_evaluation()
        print("\n✅ LangSmith evaluation with CSV export completed successfully!")
    except Exception as e:
        print(f"\n❌ LangSmith evaluation failed: {e}")