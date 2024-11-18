# In your chain invocation:
chain_result = safety_plan_chain.invoke(
    formatted_user_input,
    config={
        "tags": ["runtime_tag", "safety_plan_execution"],
        "metadata": {"execution_time": datetime.now().isoformat()}
    }
) 