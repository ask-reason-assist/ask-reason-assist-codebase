def get_prompt(propositions, task, bnf_spec, grammar_prompt, few_shot):
    prompt_with_grammar = f"""
You are an AI assistant specializing in formal methods and temporal logic. Your task is to translate a natural language requirement into a Temporal Logic (TL) formula, strictly adhering to the provided grammar.

You MUST generate the formula according to the following BNF-like grammar: 
```ebnf
{bnf_spec}
```

Return only the STL formula, without any additional text or explanation. The STL formula MUST adhere to the BNF grammar provided above.
    """
    
# Natural Language Requirement - "{task}"
# Relevant Propositions - {str(propositions)[1:-1]}
# Temporal Logic Specification - 
    
    prompt_without_grammar = f"""
You are an AI assistant specializing in formal methods and temporal logic. Your task is to translate a natural language requirement into a Temporal Logic (TL) formula.


Return only the STL formula, without any additional text or explanation.
    """

    if grammar_prompt:
        return prompt_with_grammar
    else:
        return prompt_without_grammar

def get_llama_bnf_spec(propositions):
    
    propositions = [p.replace("_", "-") for p in propositions]

    bnf_spec = f"""
root ::= ws expr ws
expr ::= term (binary-op term)*

term ::= atomic-formula | unary-op ws "(" ws expr ws ")" | unary-op ws atomic-formula | "~" ws term | "(" ws expr ws ")"

predicate-name ::= {" | ".join(f'"{p}"' for p in propositions)}
atomic-formula ::= predicate-name | "(" ws predicate-name ws ")"
ws ::= [ \t\n]*

binary-op ::= ws ("&" | "|" | "->" | "U") ws
# & (and) Both predicates are true
# | (or) Either or both predicates are true
# -> (implies) The second predicate must be true whenever the first predicate is true
# U (until) The first predicate must be true until the second predicate is true

unary-op ::= "G" | "F"
# G (globally): Predicate must always be true at every timestep
# F (eventually): Predicate must be true at some time in the future"""
    return bnf_spec
