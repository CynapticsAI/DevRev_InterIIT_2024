# important classes
import json
import hnswlib
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

# It formats the output by printing it in color.

# Defined arguments
API_KEY = "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" # api_key
model_name_tr = "gpt-3.5-turbo"  # model name
model_name_ta = "gpt-4-1106-preview"
tool_list_path = './tools.json' # list of tools path[ ]
example_path = './examples.json'  # list of examples path
zero_shot = 0
no_of_examples = 2

# ---------------------------------------------------some constants ----------------------------------------------------------------------

EF = 100  # EF
K = 3  # top k number
COSINE_THRESHOLD = 0.3  # cosine threshold
biencoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device="cpu"
)
client = OpenAI(api_key=API_KEY, timeout=60, max_retries=2)

class color:
    PURPLE = "\033[1;35;48m"
    CYAN = "\033[1;36;48m"
    BOLD = "\033[1;37;48m"
    BLUE = "\033[1;34;48m"
    GREEN = "\033[1;32;48m"
    YELLOW = "\033[1;33;48m"
    RED = "\033[1;31;48m"
    BLACK = "\033[1;30;48m"
    UNDERLINE = "\033[4;37;48m"
    END = "\033[1;37;0m"

class Tools:
    '''
      A class to represent tools.
      ...
      Attributes
      ----------
      - tools: list
        a list of tools (json format as described above)

      - examples: list
        a list of examples (json format as described above)

      Methods
      -------
      - check_json()
          checks whether the tool added is in the defined JSON schema

      - build_index()
          assigns index values to the examples and their respective tool calls to create an index list.

      - add_tool()
          adds new tool to the index list.

      - add_example()
          adds the new example in the example pool and modifies the index list.

      - modify_example()
          modifies an existing example present in the example list.

      - update_tool()
          updates an exisiting tool in the tool list.

      - replenish_examples()
          on deletion of tool replenish the examples for the tools where the num_examples < threshold

      - delete_tool()
          deletes the tool from the index list and example list.

      - similarity_retriever()
          creates the hsnw index for the example queries

    '''
    def __init__(self, tools, examples):
      # assuming the self.tools to be a dict of tools with tool name as the key and tool info as the value.
      self.tools = tools
      self.examples = examples
      self.index = {}
      self.build_index()
      self.th = 2
      self.query_embeddings = []
      self.queries = []
      self.search_index = None
      self.similarity_retriever()

    def check_json(self,tool_json):
      keys = ['argument_list', 'title', 'tool_description', 'tool_name']
      argument_keys = ['argument_description', 'argument_name', 'argument_type', 'example']
      if type(tool_json) != type({}):
        raise Exception('Given tool json is not a dictionary')
      if keys != sorted(list(tool_json.keys())):
        raise Exception(
            """Keys don't match
              Expected Keys: {}
              Given Keys: {}
            """.format(keys, sorted(list(tool_json.keys())))
        )
      if type(tool_json['argument_list']) != type([]):
          raise Exception('Given argument list is not a list')
      for idx, arg in enumerate(tool_json['argument_list']):
        if type(arg) != type({}):
          raise Exception(f'Argument at the index: {idx} is not a dictionary')
        if argument_keys != sorted(list(arg.keys())):
          raise Exception(
            """Keys don't match at {}
              Expected Keys: {}
              Given Keys: {}
            """.format(idx, argument_keys, sorted(list(arg.keys())))
          )

    def build_index(self):
        '''
        Indexes all the examples with their respective tool calls.
        For example: {'<tool_name>': {'num_examples': <no. of examples containing tool_name>, 'indices': <list of example indices containing respective tool_name>}}
        Parameters
        ----------
          None

        Returns
        ----------
          None

        Modifies
        ----------
          self.index
        '''
        for tool in self.tools:
            self.index[tool] = {'num_examples': 0, 'indices': []}
        for i, example in enumerate(self.examples):
            for tool in example['answer']:
                self.index[tool['tool_name']]['num_examples'] += 1
                self.index[tool['tool_name']]['indices'].append(i)

    def add_tool(self, tool):
        '''
        On addition of any new tool, this function gets called to add the new tool in self.index.

        Parameters
        ----------
          - tool : json (format described above)

        Returns
        ----------
          None

        Modifies
        ----------
          self.index
        '''
        # self.check_json(tool)
        tool_name = tool['tool_name']
        if tool_name in self.tools.keys():
          if tool==self.tools[tool_name]:
            print(color.YELLOW+'Already Exists!'+color.END)
          else:
            self.update_tool(tool)
            print(color.YELLOW+f"[WARNING] You tried adding a tool that already exists, so updating the tool '{tool_name}'"+color.END)
        else:
          self.tools[tool_name] = tool
          self.index[tool_name] = {'num_examples': 0, 'indices': []}
          self.add_example(tool, self.th)
          print(color.GREEN+f"[SUCCESS] Added the tool '{tool_name}'"+color.END)

    def add_example(self, tool, no_of_eg):
        '''
        On addition of new tool or replenishing examples on deletion, this function gets called to add the new example in self.examples and modifies index list accordingly.

        Parameters
        ----------
          - tool : json (format described above)
          - no_of_eg : int

        Returns
        ----------
          None

        Modifies
        ----------
          self.examples
          self.index
        '''
        tool_name = tool['tool_name']
        try:
          ex = self.examples[:3]
        except:
          ex = self.examples
        message = create_prompt_for_new_example(list(self.tools.values()), tool, ex, n=no_of_eg)
        res = client_conn(message, model_name_ta)
        res = res.choices[0].message.content
        try:
          new_examples = json.loads(res)
        except:
          new_examples = get_parsed_json(res)
        for example in new_examples:
          tool_calls = example['answer']
          for tool in tool_calls:
            new_tool_name = tool['tool_name']
            self.index[new_tool_name]['num_examples'] += 1
            self.index[new_tool_name]['indices'].append(len(self.examples))
          self.examples.append(example)
        self.similarity_retriever()

    def modify_example(self, tool):
        '''
        On modification of a tool, this function gets called to modify the examples where the tool is used.

        Parameters
        ----------
          None

        Returns
        ----------
          None

        Modifies
        ----------
          self.examples
        '''
        tool_name = tool["tool_name"]
        indices = self.index[tool_name]["indices"]
        relevant_examples = []
        for ind in indices:
            relevant_examples.append(self.examples[ind])
        message = create_prompt_for_modified_example(
            list(self.tools.values()), tool, relevant_examples
        )
        res = client_conn(message, model_name_ta)
        res = res.choices[0].message.content
        try:
          examples = json.loads(res)
        except:
          examples = get_parsed_json(res)
        for i in range(len(examples)):
            self.examples[indices[i]] = examples[i]

        print(color.GREEN + f"[SUCCESS] Modified Examples!" + color.END)
        self.similarity_retriever()

    def update_tool(self, tool):
        '''
        On modification of a tool, this function gets called update the tool.

        Parameters
        ----------
          tool : json (format described above)

        Returns
        -------
          None

        Modifies
        --------
          self.tools
        '''
        # self.check_json(tool)
        tool_name = tool['tool_name']
        if tool_name in self.tools.keys():
          self.modify_example(tool)
          self.tools[tool_name] = tool
          print(color.GREEN+f"[SUCCESS] Updated the tool '{tool_name}'"+color.END)
        else:
          print(color.YELLOW+f"[WARNING] You tried updating a tool that does not exist, so added the tool '{tool_name}'"+color.END)
          self.add_tool(tool)

    def replenish_examples(self):
        '''
        On deletion of any new tool if number of examples for other tools goes below threshold, this function gets called to create examples.

        Parameters
        ----------
          None

        Returns
        -------
          None

        Modifies
        --------
          None
        '''
        for i,tool in enumerate(self.index):
          if ((self.index[tool]['num_examples']<self.th) and (len(self.tools)>4)):
            self.add_example(self.tools[tool], self.th-self.index[tool]['num_examples'])

    def delete_tool(self, tool_name):
        '''
        On deletion of any new tool, this function gets called to delete the new tool in self.index and remove the respective examples.

        Parameters
        ----------
          - tool_name : str

        Returns
        -------
          None

        Modifies
        --------
          self.index
          self.examples
        '''
        if tool_name not in self.tools.keys():
          print(color.RED+"[ERROR] You tried deleting a tool that does not exist."+color.END)
        else:
          indices = self.index[tool_name]['indices']
          del self.index[tool_name]
          for idx, tool in enumerate(list(self.tools.values())):
              if tool['tool_name'] == tool_name:
                  del self.tools[tool_name]
                  break
          if len(indices) != 0:
              for idx, index in enumerate(indices):
                  del self.examples[index-idx]
              self.index = {}
              self.build_index()
          print(color.GREEN+f"[SUCCESS] Deleted the tool '{tool_name}'"+color.END)
          self.replenish_examples()
          self.similarity_retriever()

    def similarity_retriever(self):
        '''
        On addition of any new example, this function gets called to create the new hnsw index for queries.

        Parameters
        ----------
          None

        Returns
        -------
          None

        Modifies
        --------
          self.query_embeddings
          self.queries
          self.search_index
        '''
        self.query_embeddings, self.queries = create_example_query_embeddings(self.examples)
        self.search_index = create_hnsw_index(np.array(self.query_embeddings))

# ------------------------------------------important functions------------------------------------------------------------


def read_file(path):
    """
    parameters
    ----------
    - path: str
        path of the file to read in json.

    returns
    ---------
    - file: object
        the json file object.
    """

    with open(path, "r") as f:
        file = json.load(f)
    return file


def create_query_embedding(query):
    """
    Encodes the query to get its embedding.

    parameters
    ---------
    - query: str

    returns
    ---------
    - embedding: numpy array
      embedding of the query.

    """
    embedding = biencoder.encode([query], normalize_embeddings=True)[0]
    return embedding


def create_example_query_embeddings(examples):
    """
    creates query embeddings and saves it.

      parameters
      ----------
      - examples: list
          List of examples in json format.

      returns
      ---------
      - query_embeddings: list
          a list of query embeddings.

      - answers: list
          a list of answers corresponding to each query.

      - queries: list
          a list of queries.

    """
    query_embeddings = []
    answers = []
    queries = []
    for example in examples:
        query = example["query"]
        answer = example["answer"]
        queries.append(query)
        query_embedding = create_query_embedding(query)
        query_embeddings.append(query_embedding)
        answers.append(answer)
    np.save("query_embeddings.npy", query_embeddings)
    return query_embeddings, queries


def create_hnsw_index(embedding, M=16, efC=100):
    """
    creates the HNSW index.

    parameters
    ----------
    - embedding: list
      query embedding

    - M: int
      default = 16

    - efc: int
      default = 100

    returns
    ----------
    - index: object
    """
    embeddings = embedding
    num_dim = embeddings.shape[1]
    ids = np.arange(embeddings.shape[0])
    index = hnswlib.Index(space="ip", dim=num_dim)
    index.init_index(max_elements=embeddings.shape[0], ef_construction=efC, M=M)
    index.add_items(embeddings, ids)
    return index


def find_nearest_neighbors(query_embedding, queries, search_index):
    """
    Finds the k nearest neighbors using cosine similarity.
    parameters
    ----------
    - query_embedding: list
      the query embedding whose similarity check has to be made.

    - queries: list
      a list of queries in the examples.

    - search_index: object

    returns
    ----------
    - query_list: list
      a list of similar queries.
    """
    search_index.set_ef(EF)
    labels, distances = search_index.knn_query(
        query_embedding, k=K
    )  # Find the k-nearest neighbors for the query embedding
    labels = [
        label
        for label, distance in zip(labels[0], distances[0])
        if (1 - distance) >= COSINE_THRESHOLD
    ]
    query_list = [queries[i] for i in labels]
    return query_list


def rerank_queries_with_cross_encoder(query, chunks):
    """
    Sorts the chunks based on their scores in descending order.
    parameters
    ---------
      - query: str
      - chunks: list
          the list of similar chunks
    returns
    ---------
      - sorted_chunks: list
          the response after reranking.
    """
    pairs = [(query, chunk) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
    return sorted_chunks


def rerank_queries_with_cross_encoder(query, chunks):
    """
    Sorts the chunks based on their scores in descending order.
    parameters
    ---------
      - query: str
      - chunks: list
          the list of similar chunks
    returns
    ---------
      - sorted_chunks: list
          the response after reranking.
    """
    pairs = [(query, chunk) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
    return sorted_chunks


# -------------------------------------------fetch topk examples--------------------------------------------------------------


def get_index(queries, topk_queries):
    """
    Fetches the topk_indices given topk queries.

    parameters:
      - queries: list
          A list of all queries.
      - topk_queries: list
          A list of topk queries.

    returns:
      - index: list of int
    """
    index = []
    for i in topk_queries:
        index.append(queries.index(i))
    return index


def get_topk_examples(topk_index, examples):
    """
    Fetches topk examples from the examples given indices of topk examples.
    parameters
    ----------
    - topk_index: int
        list of indexes of topk tools.

    - examples: list
        list of all examples.

    returns
    ----------
    - res: list
        list of topk examples.
    """

    res = []
    for index in topk_index:
        res.append(examples[index])
    return res


def get_topk_given_query(query, queries, search_index, examples):
    '''
    fetches topk examples given the queries.
    parameters
    ----------
      - query: str
          user query

      - queries: list
          a list of all queries.

      - search_index:
          index object

      - examples: list
          a list of all examples.
    returns
    -----------
      - topk_examples: list
          a list of topk examples.
    '''
    query_embedding = create_query_embedding(query)
    topk_queries = find_nearest_neighbors(query_embedding, queries, search_index)
    ranked_topk_queries = rerank_queries_with_cross_encoder(query, topk_queries)
    topk_indices = get_index(queries,ranked_topk_queries)
    topk_examples = get_topk_examples(topk_indices, examples)
    return topk_examples


# ---------------------------------------------------prompt generation functions-------------------------------------------------
def client_conn(message, model_name):
    '''
      Generates an API call instant.
      parameters
      ----------
        - message: list
            A list of conversation between ChatGPT and user.

      returns
      ----------
        - completion: object
            An API instant.
    '''
    # please don't change the timeout as example generation is time consuming
    client = OpenAI(api_key=API_KEY, timeout=120, max_retries=2)
    completion = client.chat.completions.create(
        model=model_name,
        messages=message,
        temperature=0.2
    )
    return completion

def create_prompt_for_query(user_query, examples, tools):
    message = [
        {
            "role": "system",
            "content": "The following is a friendly conversation between a human and an AI. The AI is professional and parses user input to several tasks. If the AI does not know the answer to a question, it truthfully says it does not know. The AI will be provided with a set of tools their descriptions and the argument in them. Here is the list of tools:"+ json.dumps(tools) + " \n Provide the answer in the exact format as given in the following examples. \nExamples"
        }
    ]
    for example in examples:
        query = example['query']
        answer = example['answer']
        user_prompt = "Query: "+ str(query)
        assistant_prompt = str(answer)
        message.append(
        {
            "role" : "user",
            "content": user_prompt
        })
        message.append(
        {
            "role" : "assistant",
            "content": assistant_prompt
        })

    message.append(
    {
        "role":"user",
        "content":"Use the above tools to learn how to use the tool on any query. Analyse how to parse the query and extract the correct information and place in the argument name and value. Use all the required tools and arguments in correct order of its calling based on the query and your learning from all the examples. Do not assume any value, you can take the value from query or the previous called tool as shown in the examples. Also focus on the allowed values argument present in tool definition."
    })
    message.append({
        "role" : "user",
        "content" : "Now its your task to respond to the user queries in the same format as that in the above examples which is json."
    })
    message.append({
        "role" : "user",
        "content" : "Query: "+ user_query
    })
    message.append(
    {
        "role":"system",
        "content": "Generate the answer in a json format only. Enclose the strings in double quotes"
    })

    return message


def create_prompt_zero_shot(user_query, tools):
    '''
      Creates the prompt given user query, single examples and multi examples.
      parameters
      ----------
        - user_query: str
          the user query.

        - single_example: list
          the list of single tool examples.

        - multi_example: list
          the list of multi tiik examples.

      returns
      ----------
        - message: list
          the list of user prompts.
    '''
    message = [
        {
            "role": "system",
            "content": "You are a intelligent AI agent specialized in giving the tool responses given a dictionary of tools. Here is the dictionary of tools: "+ json.dumps(tools)
        }
    ]
    message.append({
        "role" : "user",
        "content" : '''
        Now its your task to respond to the user queries in the format given below
        FORMAT:[{"tool_name": "...", "arguments": [{"argument_name": "...", "argument_value": ... (depending on the argument_type)}, ...]}, ...]
        To reference the value of the ith tool in the chain, use $$PREV[i] as argument value. i = 0, 1, .. j-1; j = current tool’s index in the array If the query could not be answered with the given set of tools, output an empty list instead.
        Output in the JSON format
        '''
    })
    message.append({
        "role" : "user",
        "content" : "Query: "+ user_query
    })
    return message


def create_prompt_for_modified_example(old_tools, modified_tool, relevant_examples):
  message = [
      {
          "role":"system",
          "content":"You are an intelligent AI Agent specialized in modifying the old data and generating the new relevant data."
      }
  ]
  message.append({
      "role":"user",
      "content":"Given a list of old tools : " + json.dumps(old_tools) + "Let us say that I modified the tool" + "'" + modified_tool['tool_name'] + "'" + "to be" + json.dumps(modified_tool)+"""
      Now your task is to modify the following examples where this tool was used according to its new definition keeping in mind the new schema of json mentioned above.
      """ + json.dumps(relevant_examples)
  })
  message.append({
      "role":"system",
      "content":"Your response should be in json format only with the strings enclosed in double quotes ready to go in json.loads"
  })
  return message
# ---------------------------------------------------------------------------------------------------------------------------


def create_prompt_for_new_example(old_tools, new_tool, examples, n=no_of_examples):
  message = [
      {
          "role":"system",
          "content":"You are an intelligent AI Agent specialized in generating the new relevant data."
      }
  ]
  message.append({
      "role":"user",
      "content":"Given a list of old tools : " + json.dumps(old_tools) + "and a new tool" + "'" + new_tool['tool_name'] + "'" + " to be " + json.dumps(new_tool)+"""
      Now your task is to create """+str(n)+""" examples of the usage of the new tool along with any of the tools from the old tool list, similar to the following example:.
      """ + json.dumps(examples)
  })
  message.append({
      "role":"system",
      "content":"Your response should be in json format with the strings enclosed in double quotes. Note that To reference the value of the ith tool in the chain, use $$PREV[i] as argument value. i = 0, 1, .. j-1; j = current tool’s index in the array If the query could not be answered with the given set of tools, output an empty list instead."
  })
  return message
# -------------------------------------------------retriever---------------------------------------------------------------------
def create_tool_str(tools):
  tool_str = ''
  for tool in tools:
      tool_str += f"Tool: {tool['tool_name']}, Desc: {tool['tool_description']}\n"
  return tool_str

def tool_retriever_prompt(tool_str,user_query):
    message=[
      {"role": "system", "content": "You are an intelligent assistant. Please help the user below."},
      {"role": "user", "content": f'''You are given the following set of tools:\n {tool_str} \n Can you please figure out which tools the query "{user_query}" will require to solve, out of these tools? Please return only the tool names inside []. If it does not need any tool, return an empty list'''}]
    return message


def tool_retriever(tools, user_query):
  tool_str = create_tool_str(tools)
  message = tool_retriever_prompt(tool_str, user_query)
  output = client_conn(message, model_name_tr)
  output = output.choices[0].message.content
  list_delim=output[output.find('[')+1:output.find(']')]
  tools_retrieved = set()
  for i in list_delim.split(','):
    if (i.strip()!=''):
      tools_retrieved.add(i.strip().replace('\'','').replace('\"',''))
  return tools_retrieved


def final_tools(tools, tool_names):
  tool = []
  for tool_name in tool_names:
    tool.append(tools[tool_name])
  for tool_check in list(tools.values()):
    if not tool_check["argument_list"]:
      tool.append(tool_check)
  return tool


# -------------------------------------------------------bonus section prompts---------------------------------------------------------------
def create_prompt_for_query_bonus(user_query, examples, tools):
    message = [
        {
            "role": "system",
            "content": f"""The following is a friendly conversation between a human and an AI.
             The AI is professional and parses user input to several tasks. If the AI does not
             know the answer to a question, it truthfully says it does not know. The AI will be
              provided with a set of tools their descriptions and the argument in them. Here is
               the list of tools:"+ {json.dumps(tools)} + "\n Provide the answer in the
                exact format as given in the following examples. \nExamples """
        }
    ]
    for example in examples:
        query = example['query']
        answer = example['answer']
        user_prompt = "Query: "+ str(query)
        assistant_prompt = str(answer)
        message.append(
        {
            "role" : "user",
            "content": user_prompt
        })
        message.append(
        {
            "role" : "assistant",
            "content": assistant_prompt
        })

    message.append(
    {
        "role":"user",
        "content":"Use the above tools to learn how to use the tool on any query. Analyse how to parse the query and extract the correct information and place in the argument name and value. Use all the required tools and arguments in correct order of its calling based on the query and your learning from all the examples. Do not assume any value, you can take the value from query or the previous called tool as shown in the examples. Also focus on the allowed values argument present in tool definition."
    })
    message.append(
    {
        "role":"user",
        "content":f"After producing the list of tools, analyze the query and figure out whether it requires the combination of tool outputs via mathematical operations, iterations, conditional logic etc. or not. In case it does, use the lambda function to produce the required results. Examples of such queries are given below: \n "
    })

    message.append(
        {
            "role":"user",
            "content":f"Find all tasks created by user 'USER-321' and check if there are more than 10 such tasks"
        }
    )
    message.append(
        {
            "role":"assistant",
            "content":"""
            "answer": [
        {
          "tool_name": "search_object_by_name",
          "arguments": [
            {
              "argument_name": "query",
              "argument_value": "USER-321"
            }
          ]
        },
        {
          "tool_name": "works_list",
          "arguments": [
            {
              "argument_name": "created_by",
              "argument_value": [
                "$$PREV[0]"
              ]
            },
            {
              "argument_name": "type",
              "argument_value": [
                "task"
              ]
            }
          ]},
          {
          "tool_name": "lambda",
          "arguments": [
            {
              "argument_name": "expression",
              "argument_value": "lambda $$PREV[1]: True if len($$PREV[1]) > 10 else False"
            }
          ]
        }
      ]
            """
        }
    )

    message.append({
        "role" : "user",
        "content" : "Now its your task to respond to the user queries in the same format as that in the above examples which is json. Use the lambda function only when necessary."
    })
    message.append({
        "role" : "user",
        "content" : "Query: "+ user_query
    })
    message.append(
    {
        "role":"system",
        "content": "Generate the answer in a json format only. Enclose the strings in double quotes"
    })

    return message



# -------------------------------------------------miscellaneous functions------------------------------------------------------------
def create_tool_dict(tools):
    tool_dict = {}
    for i in tools:
        tool_dict[i["tool_name"]] = i
    return tool_dict

def get_parsed_json(text_parsed):
  ans_list = []
  json_str = text_parsed.split('```json')
  for i in json_str:
    if "```" in i:
      json_data = i.split("```")[0].strip()
      if json_data:  # Check if the JSON data is not empty
        json_obj = json.loads(json_data)
        if type(json_obj)==list:
          ans_list.extend(json_obj)
        else:
          ans_list.append(json_obj)
  return ans_list
# -------------------------------------------------postprocessing---------------------------------------------------------------------
def get_json(pred):
  '''
  parameters
  -----------------
  pred: str
    Answer predicted by the LLM as a string

  returns
  -----------------
  json_pred: list
    List of dictionaries that represents the input string as a json
  '''

  try:
    # Tries to find ```json ``` type json format
    return json.loads(pred[pred.find('```json'):-1*("".join(reversed(pred)).find('```')+1)])
  except:
    # Tries to find first instance of '[' from the left and first instance of ']' from the right, and converts all in between ito a json.
    try:
      return json.loads(pred[pred.find('['):-1*("".join(reversed(pred)).find(']')+1)] + ']')
    except:
      # Tries to fix keys/ strings being wrapped in single quotes, then tries to decode as above
        pred= pred.replace('\'','\"')
        try:
          return json.loads(pred[pred.find('['):-1*("".join(reversed(pred)).find(']')+1)] + ']')
          # Tries to check for instances of boolean values true and false, that might have been misspelt as True and False
        except:
          pred = pred.replace("True", "true").replace("False", "false")
          return json.loads(pred[pred.find('['):-1*("".join(reversed(pred)).find(']')+1)] + ']')

def dict_unwrap(json_pred):
  # Unwraps a dictionary if GPT-4 outputs one instead of a list
  if type(json_pred)==dict:
    for key in json_pred.keys():
      if type(json_pred[key])==list:
        print(json_pred[key])
        return json_pred[key]
    return []
  return json_pred

def list_in_str_handler(json_pred):
  '''
  parameters
  -----------------
  json_pred: list
    List of dictionaries that represents the tool call sequence

  -----------------
  json_pred: list
    Tool call sequence with string arguments like '[arg_val]' turned into 'arg_val'
  '''
  # Iterate over tool call sequence and get argument values
  for i, tool in enumerate(json_pred):

    for j, arg in enumerate(tool["arguments"]):
      arg_val= arg["argument_value"]
      # If value is a string and it starts with [ and ends with ], remove them
      if (type(arg_val)==str):
        if arg_val.startswith('[') and arg_val.endswith(']'):
          arg["argument_value"]=arg_val[1:-1]
      # If value is a list, then iterate over it and perform similar operations as above
      elif (type(arg_val)==list):
        for num_item,arg_val_item in enumerate(arg_val):
          if (type(arg_val_item)==str):
            if arg_val_item.startswith('[') and arg_val_item.endswith(']'):
              arg["argument_value"][num_item]=arg_val_item[1:-1]
  return json_pred

def func_name_handler(json_pred,tools,no_arg_tool_list):
  '''
  parameters
  -----------------
  json_pred: list
    List of dictionaries that represents the tool call sequence

  tools: dict
    Dict representing tools

  no_arg_tool_list: list
    List of tool names that do not have arguments
  returns
  -----------------
  json_pred: list
    Tool call sequence with arguments with $${function_name} errors removed
  '''
  # Iteration variable
  i=0
  # WHILE loop is required, len(range()) does not work as the loop condition is kept static while items are inserted into the loop.
  while(i<len(json_pred)):
    tool=json_pred[i]
    for j, arg in enumerate(tool["arguments"]):
      if arg["argument_name"] in tools[tool["tool_name"]]["args"] :
          # Check argument value
          temp = arg["argument_value"]
          if type(temp)==str:
            # If argument value starts with $$ but is not $$PREV[i]
            if (temp.startswith('$$')) and not temp.startswith('$$PREV['):
              # Remove the $$
              temp_lowercase_call=temp.lower()[2:]
              # Iterate over tools with no arguments to figure the appropriate tool to call
              for no_arg_tool in no_arg_tool_list:
                if temp_lowercase_call.startswith(no_arg_tool):
                  # Create tool call
                  tool_ins = {}
                  tool_ins['arguments']=[]
                  tool_ins['tool_name']=no_arg_tool
                  # Insert tool into the list
                  json_pred.insert(i,tool_ins)
                  i+=1
                  # Iterate over the tools, starting from the tool under consideration
                  for i_n, tool_n in enumerate(json_pred[i:]):
                    for j_n, arg_n in enumerate(tool_n["arguments"]):
                      # Check the argument value
                      prevset = arg_n["argument_value"]
                      # If the argument value is
                      if (type(prevset)==str):
                        # If the argument is of $$PREV[i] type, and it referenced the returned value of the tool that
                        # came at the same position as, or after the inserted tool, increment i for it,
                        if prevset.startswith("$$PREV["):
                          try:
                            n=int(prevset[7:-1])
                          except:
                            indexing_pos=prevset.find('][')
                            try:
                              n=int(prevset[7:indexing_pos])
                            except:
                              pass
                        if n>=i-1:
                          arg_n["argument_value"]=f"$$PREV[{n}]"
                      # If the argument value is a list, iterate over the values and perform the same process as above.
                      elif type(prevset)==list:
                        for list_num,prev_val in enumerate(prevset):
                          if prev_val.startswith("$$PREV["):
                            n=0
                            try:
                              n=int(prevset[7:-1])
                            except:
                              indexing_pos=prevset.find('][')
                              try:
                                n=int(prevset[7:indexing_pos])
                              except:
                                pass
                          if n>=i-1:
                            arg_n["argument_value"][list_num]=f"$$PREV[{n}]"
                  arg["argument_value"]=f"$$PREV[{i-1}]"
          # otherwise the argument value is a list, iterate over this list, and perform the same operations as above.
          elif (type(temp)==list):
            for num_arg,temp_el in enumerate(temp):

              if type(temp_el)==str:
                if (temp_el.startswith('$$')) and not temp_el.startswith('$$PREV'):
                  temp_lowercase_call=temp_el.lower()[2:]

                  for no_arg_tool in no_arg_tool_list:
                    if temp_lowercase_call.startswith(no_arg_tool):
                      tool_ins = {}
                      tool_ins['arguments']=[]
                      tool_ins['tool_name']=no_arg_tool
                      json_pred.insert(i,tool_ins)
                      i+=1

                      for i_n, tool_n in enumerate(json_pred[i:]):
                        for j_n, arg_n in enumerate(tool_n["arguments"]):

                          prevset = arg_n["argument_value"]
                          if (type(prevset) not in [list,bool,float]):
                            if prevset.startswith("$$PREV[") and int(prevset[7:-1])>=i-1:
                              n=int(prevset[7:-1])+1
                              arg_n["argument_value"]=f"$$PREV[{n}]"

                          elif type(prevset) not in [bool,float]:
                            for list_num,prev_val in enumerate(prevset):
                              try:
                                if prev_val.startswith("$$PREV[") and int(prev_val[7:-1])>=i-1:
                                  n=int(prevset[7:-1])+1
                                  arg_n["argument_value"][list_num]=f"$$PREV[{n}]"

                              except:
                                pass
                      arg["argument_value"][num_arg]=f"$$PREV[{i-1}]"
    i+=1
  return json_pred



def unknown_tool_remover(json_pred,tools):
    for i,tool_call in enumerate(json_pred):
        if tool_call["tool_name"] not in tools.keys():
            return True


def type_handler(json_pred,tools,array_check,string_check,num_check,bool_check,string_to_boolean):
  '''
  parameters
  -----------------
  json_pred: list
    List of dictionaries that represents the tool call sequence

  tools: dict
    Dict representing tools

  array_check: list
    List of keywords to check for array return types

  string_check: list
    List of keywords to check for string return types

  num_check: list
    List of keywords to check for numeral return types

  bool_check: list
    List of keywords to check for boolean return types

  string_to_boolean: dict
    Dictionary mapping common representations of True/False values to booleans

  returns
  -----------------
  json_pred: list
    Tool call sequence with tool inputs respecting tool input type requirements
  '''
  for i, tool in enumerate(json_pred):

    for j, arg in enumerate(tool["arguments"]):

      if arg["argument_name"] in tools[tool["tool_name"]]["args"] :

          arg_type = tools[tool["tool_name"]]["args"][arg["argument_name"]]["argument_type"]

          temp = json_pred[i]["arguments"][j]["argument_value"]

          # Split the argument type by spaces for easier checks later
          typcheck = set(arg_type.lower().split(' '))
          # To check if arg_type is supposed to be a list, but it is not
          if typcheck.intersection(array_check) and type(temp)!=list:
            # If argument is not $$PREV type and is a string, convert to array of strings

            if not temp.startswith("$$") and typcheck.intersection(string_check) :
              temp = [str(temp)]
            # If argument type has integer, convert to list of integers
            elif typcheck.intersection(num_check):
              if type(temp) in [int,float]:
                temp = [temp]
              else:
                try:
                  temp = [int(temp)]
                except:
                  pass
            # If argument type has boolean, convert to list of booleans
            elif typcheck.intersection(bool_check):

              try:
                if type(temp)==str:
                  temp = [string_to_boolean.get(temp,temp)]
                else:
                  temp=[bool(temp)]
              except:
                pass
          # If argument type is string, convert to string
          elif typcheck.intersection(string_check) and type(temp) != str:
            # If the argument type is currently a list, convert to string. Only the first argument is going to be considered.
            if (type(temp)==list and not temp):
              try:
                if (not temp[0].startswith("$$")):
                    temp = str(temp[0])
              except:
                pass
            if (type(temp) in [int,float]):
              temp=str(temp)
          # If argument type is boolean, convert to boolean
          elif typcheck.intersection(bool_check) and type(temp)!= bool :
            # If the argument type is currently a list, convert to boolean. Only the first argument is going to be considered.
            if (type(temp)==list and temp):
              if (type(temp[0])==str):
                if not temp[0].startswith("$$"):
                  try:
                    temp = string_to_boolean.get(temp[0], temp)
                  except:
                    pass
              elif (type(temp[0])==bool):
                temp=temp[0]
              elif (type(temp[0]) in [int,float]):
                temp=bool(temp[0])
            # If the argument type is currently a string, convert to boolean
            elif type(temp)==str:
              temp=string_to_boolean.get(temp,temp)
          json_pred[i]["arguments"][j]["argument_value"] = temp

      else:
        print("arg name not match for",arg)
  return json_pred


def prev_ret_type_handler(json_pred,tools,array_check,string_check,num_check,bool_check):
  '''
  parameters
  -----------------
  json_pred: list
    List of dictionaries that represents the tool call sequence

  tools: dict
    Dict representing tools

  array_check: list
    List of keywords to check for array return types

  string_check: list
    List of keywords to check for string return types

  num_check: list
    List of keywords to check for numeral return types

  bool_check: list
    List of keywords to check for boolean return types

  returns
  -----------------
  json_pred: list
    Tool call sequence with $$PREV[i] type arguments respecting tool input type requirements
  '''
  # Iterate over list of dictionaries
  for i, tool in enumerate(json_pred):
    for j, arg in enumerate(tool["arguments"]):

      if arg["argument_name"] in tools[tool["tool_name"]]["args"] :
          # Fetch argument type, and the current argument value
          arg_type = set(tools[tool["tool_name"]]["args"][arg["argument_name"]]["argument_type"].split(' '))
          temp = json_pred[i]["arguments"][j]["argument_value"]
          # If current argument is a string
          if type(temp)== str:
            # If it is a $$PREV type argument
            if temp.startswith('$$PREV'):
              # Identify the return type of the tool call referenced by $$PREV[i],
              refer=temp[7:-1]
              ref_type= set(tools[json_pred[int(refer)]["tool_name"]]["return_type"].split(' '))
              # If the referenced tool does not return an array, but the argument expects one
              if not ref_type.intersection(array_check) and arg_type.intersection(array_check):
                json_pred[i]["arguments"][j]["argument_value"]=[temp]
          # Otherwise, if the argument value is a list, iterate over it
          elif (type(temp)==list):
            for temp_el in temp:
              # Perform similar actions as above
              if type(temp_el)==str:
                if temp_el.startswith('$$PREV'):
                  refer=temp_el.lower()[7:-1]
                  ref_type= set(tools[json_pred[int(refer)]["tool_name"]]["return_type"].split(' '))
                  if (ref_type.intersection(array_check) and arg_type.intersection(array_check)) or (not ref_type.intersection(array_check) and not arg_type.intersection(array_check)):
                    json_pred[i]["arguments"][j]["argument_value"]=temp_el

  return json_pred

def postprocess(json_pred, tool_data):
  '''
  parameters
  -----------------
  json_pred: str
    Answer predicted by the LLM as a string

  tool_data: dict
    dictionary representing the given tools

  returns
  -----------------
  json_pred: list
    List of dictionaries that represents the final answer
  '''
  # turn json in string format into list of dicts
  # Get dictionary of tools to simplify work
  json_pred = dict_unwrap(json_pred)
  tools = {}
  for i,tool in enumerate(tool_data):
    tools[tool["tool_name"]] = tool
    tools[tool["tool_name"]]["args"] = {}
    tools[tool["tool_name"]]["return_type"] = tool["return_type"]
    for arg in tool["argument_list"]:
      tools[tool["tool_name"]]["args"][arg["argument_name"]] = arg
  for tool in json_pred:
    if not tool.get('arguments',None):
      tool["arguments"] = []
  # Lists holding keywords to search for in the argument types/ return types
  array_check = ["array","list","arrays","lists"]
  string_check=["string","str","strings"]
  num_check=["integer","int32","number","float","double","float32"]
  bool_check=["bool","boolean","true","false"]
  # Dictionary mapping common strings to boolean values
  string_to_boolean = {"True": True, "False": False, "1": True, "0": False, "yes": True, "no": False, "true" : True, "false": False, 'True':True, 'False':False}

  # Get tools for which no argument is required, to fix $${function_name} errors encountered
  no_arg_tool_list = []
  for i,tool in enumerate(tool_data):
    if not tool["argument_list"]:
      no_arg_tool_list.append(tool["tool_name"])

  try:
    if unknown_tool_remover(json_pred, tools):
      return []
  except:
    pass
  try:
    json_pred = list_in_str_handler(json_pred)
  except:
    pass

  try:
  # Fix type errors
    json_pred = type_handler(json_pred,tools,array_check,string_check,num_check,bool_check,string_to_boolean)
  except:
    pass
  try:
  # Fix $${function_name} errors
    json_pred = func_name_handler(json_pred,tools,no_arg_tool_list)
  except:
    pass
  try:
  # Fix return types for $$PREV[i] type arguments
    json_pred = prev_ret_type_handler(json_pred,tools,array_check,string_check,num_check,bool_check)
  except:
    pass

  return json_pred

tools = read_file(tool_list_path)
examples = read_file(example_path)
tool_dict = create_tool_dict(tools["tools"])
tool_obj = Tools(tool_dict, examples)

def main(query):
  if len(tool_obj.examples)<5:
    message = create_prompt_zero_shot(query,tool_obj.tools)
    res = client_conn(message, model_name_ta)
    answer = []
    try:
      answer = json.loads(res.choices[0].message.content)
      answer = postprocess(answer, list(tool_obj.tools.values())) # postprocessing
    except:
      pass
    return answer

  topk_examples = get_topk_given_query(query, tool_obj.queries, tool_obj.search_index, tool_obj.examples)
  reduced_tools = tool_retriever(list(tool_obj.tools.values()), query)
  if not reduced_tools:
    return []
  tool_list = final_tools(tool_obj.tools, list(tool_obj.tools.keys()))
  if 'lambda' in reduced_tools:
    message = create_prompt_for_query_bonus(query, topk_examples, tool_list)
  else:
    message = create_prompt_for_query(query, topk_examples, tool_list)
  res = client_conn(message, model_name_ta)
  print("GPT4 Response: ",res.choices[0].message.content)
  answer = []
  try:
    answer = json.loads(res.choices[0].message.content)
    answer = postprocess(answer, list(tool_obj.tools.values())) # postprocessing
  except:
    pass
  return answer