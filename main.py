# Importing the libraries
import json
import time
import hnswlib
import numpy as np
import streamlit as st
from openai import OpenAI
import streamlit_nested_layout
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils import *

# -----------------------------------------------------------defined arguments-----------------------------------------------------------
count = 0
save = 0


# Defined arguments
API_KEY = "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"  # api_key
model_name = "gpt-3.5-turbo"  # model name
tool_list_path = "./tools.json"  # list of tools path
example_path = "./examples.json"  # list of examples path
user_query = "Summarize my P1 issues in triage"
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

# ------------------------------------------------------------------------------------------------------------------------------

if "n_args" not in st.session_state:
    st.session_state["n_args"] = 0

if "new_tool" not in st.session_state:
    st.session_state["new_tool"] = {}

if "tools" not in st.session_state:
    st.session_state.tools = list(tool_obj.tools.values())


def warn(message):
    st.toast(message)

# Save Tool
def save_tool(tool_list):
    if type(tool_list) == list:
        for t in tool_list:
            tool_obj.add_tool(t)
            st.toast("Updated", icon="✅")
    else:
        tool_obj.add_tool(tool_list)

# Function to Add Arguments
def add_arguments(i):
    global count
    with st.expander(f"Argument {i+1}"):
        st.session_state["new_tool"]["argument_list"][i][
            "argument_name"
        ] = st.text_input("Argument Name", "", key=count)
        count += 1
        st.session_state["new_tool"]["argument_list"][i][
            "argument_description"
        ] = st.text_input("Argument Description", "", key=count)
        count += 1
        st.session_state["new_tool"]["argument_list"][i][
            "argument_type"
        ] = st.text_input("Argument Type", "", key=count)
        count += 1
        st.session_state["new_tool"]["argument_list"][i]["example"] = st.text_input(
            "Argument Examples", "", key=count
        )
        count += 1
        if st.button(
            "Delete Argument", key=count, use_container_width=True, type="primary"
        ):
            warn(
                f'Deleted {st.session_state["new_tool"]["argument_list"][i]["argument_name"]}'
            )
            st.session_state["n_args"] -= 1
            del st.session_state["new_tool"]["argument_list"][i]
            st.rerun()

        count += 1

# -----------------------------------------------------------User Interface-----------------------------------------------------------
with st.sidebar:
    with st.expander("Add Tool"):
        st.session_state["new_tool"]["tool_name"] = ""
        st.session_state["new_tool"]["tool_description"] = ""
        st.session_state["new_tool"]["return_type"] = ""
        st.session_state["new_tool"]["tool_name"] = st.text_input("Tool Name", st.session_state["new_tool"]["tool_name"])
        st.session_state["new_tool"]["tool_description"] = st.text_input(
            "Tool Description", ""
        )
        st.session_state["new_tool"]["return_type"] = st.text_input(
            "Return Datatype", ""
        )
        if "argument_list" not in st.session_state["new_tool"]:
            st.session_state["new_tool"]["argument_list"] = []

        with st.expander("Arguments"):
            for i in range(st.session_state["n_args"]):
                add_arguments(i)
        cols = st.columns(2)
        with cols[0]:
            if st.button("Add", use_container_width=True):
                st.session_state["n_args"] += 1
                st.session_state["new_tool"]["argument_list"].append(
                    {
                        "argument_name": "",
                        "argument_description": "",
                        "argument_type": "",
                        "example": "",
                    }
                )
                st.rerun()

        with cols[1]:
            if st.button("Save", use_container_width=True):
                save = 1
                try:
                    # Make Sure all fields are filled
                    print(st.session_state.new_tool)
                    if not st.session_state["new_tool"]["tool_name"]:
                        raise Exception("Empty Tool Name")
                    if not st.session_state["new_tool"]["tool_description"]:
                        raise Exception("Empty Tool Description")
                    if not st.session_state["new_tool"]["return_type"]:
                        raise Exception("Empty Return Type")
                    for arg in st.session_state["new_tool"]["argument_list"]:
                        if not arg["argument_name"]:
                            raise Exception("No argument name given")
                        if not arg["argument_description"]:
                            raise Exception("No argument description given")
                        if not arg["argument_type"]:
                            raise Exception("No argument type given")
                        if not arg["example"]:
                            raise Exception("No example of argument given")
                    with st.spinner("Adding..."):
                        
                        save_tool(st.session_state["new_tool"])
                        st.session_state.tools = list(tool_obj.tools.values())
                    time.sleep(2)
                    st.session_state["new_tool"] = {}
                    st.session_state["n_args"] = 0
                except Exception as e:
                    st.toast(e)
                st.rerun()

    with st.expander("Add Tools Via Json"):
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = json.load(uploaded_file)
            # st.write(data)
            try:
                try:
                    for d in data:
                        tool_obj.check_json(d)
                except Exception as e:
                    st.toast(f"Inccorect Format: {e}")
                with st.spinner("Adding..."):
                    save_tool(data)
                    # st.session_state["tools"] += data
                    st.session_state.tools = list(tool_obj.tools.values()) 
                st.toast(":green[Added Tools!]")
            except:
                st.toast("Incorrect Format Specified")

    with st.expander("Current Tools"):
        for i, tool in enumerate(st.session_state.tools):
            with st.expander(tool["tool_name"]):
                count += 1
                st.session_state["tools"][i]["tool_description"] = st.text_input(
                    "Tool Description",
                    st.session_state["tools"][i]["tool_description"],
                    key=count,
                )
                count += 1
                with st.expander("Argument List"):
                    for j, arg in enumerate(tool["argument_list"]):
                        with st.expander(arg["argument_name"]):
                            st.session_state["tools"][i]["argument_list"][j][
                                "argument_name"
                            ] = st.text_input(
                                "Argument Name",
                                st.session_state["tools"][i]["argument_list"][j][
                                    "argument_name"
                                ],
                                key=count,
                            )
                            count += 1
                            st.session_state["tools"][i]["argument_list"][j][
                                "argument_description"
                            ] = st.text_input(
                                "Argument Description",
                                st.session_state["tools"][i]["argument_list"][j][
                                    "argument_description"
                                ],
                                key=count,
                            )
                            count += 1
                            st.session_state["tools"][i]["argument_list"][j][
                                "argument_type"
                            ] = st.text_input(
                                "Argument Type",
                                st.session_state["tools"][i]["argument_list"][j][
                                    "argument_type"
                                ],
                                key=count,
                            )
                            count += 1
                            st.session_state["tools"][i]["argument_list"][j][
                                "example"
                            ] = st.text_input(
                                "Argument Example",
                                st.session_state["tools"][i]["argument_list"][j][
                                    "example"
                                ],
                                key=count,
                            )
                            count += 1
                            columns = st.columns(2)
                            with columns[0]:
                                if st.button(
                                    "Delete",
                                    key=count,
                                    use_container_width=True,
                                    type="primary",
                                ):
                                    with st.spinner("Deleting..."):
                                        del st.session_state["tools"][i]["argument_list"][j]
                                        save_tool(st.session_state["tools"][i])
                                        st.session_state.tools = list(tool_obj.tools.values())
                                    st.toast(f":red[Deleted Tool]")
                                    st.rerun()
                                count += 1

                            with columns[1]:
                                if st.button(
                                    "Save", key=count, use_container_width=True
                                ):
                                    save_tool(st.session_state["tools"][i])
                                    st.session_state.tools = list(tool_obj.tools.values())
                                    st.rerun()
                                count += 1

                cols = st.columns([2, 2, 1])
                with cols[0]:
                    if st.button("Add", key=count, use_container_width=True):
                        st.session_state.tools[i]["argument_list"].append(
                            {
                                "argument_name": "",
                                "argument_description": "",
                                "argument_type": "",
                                "example": "",
                            }
                        )
                        st.rerun()
                    count += 1
                with cols[1]:
                    if st.button("Save", key=count, use_container_width=True):
                        with st.spinner("Wait"):
                            save_tool(st.session_state["tools"][i])
                            st.session_state.tools = list(tool_obj.tools.values())
                        st.toast(f":green[Modified Tool]")
                        st.rerun()
                    count += 1

                with cols[2]:
                    if st.button("❌", key=count):
                        warn(f"Deleted {tool['tool_name']}")
                        tool_obj.delete_tool(tool["tool_name"])
                        st.session_state["tools"] = list(tool_obj.tools.values())
                        st.rerun()
                    count += 1

from time import sleep
st.markdown("# ToolMaster")


prompt = st.chat_input("Say something")
if prompt:
    st.chat_message("user").write(prompt.replace(":", "\:"))
    with st.spinner('Querying...'):
        answer = main(prompt)
    st.chat_message("assistant").write(answer)
