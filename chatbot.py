# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from openai import OpenAI
import pickle
import numpy as np
import os
from process import predict_emotion, process_prompt


# DESIGN implement changes to the standard streamlit UI/UX

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

def classify(data):
    
    with open('../models/classifierV0', 'rb') as f:
        loaded_rf = pickle.load(f)
        preds = loaded_rf.predict(data)
        neg_score = sum(np.where(preds==0, 1, 0))
        neutral_score = sum(np.where(preds==1, 1, 0))
        pos_score = sum(np.where(preds==2, 1, 0))
    
    
#on FIRST CHAT, append primer to beginning of prompt
f = open('./chatbot-primer.txt')
primer = f.read()
f.close()


def run():
    st.set_page_config(
        page_title="EmpathEEG",
        page_icon="img/brain.png",
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "labels" not in st.session_state:
        st.session_state.labels = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.write("# Welcome to EmpathEEG! 💭")


    st.markdown(
        """
        """
    )
    st.subheader('\nWhat are you thinking about?\n')
    if prompt := st.chat_input("What's up?"):
        label = predict_emotion()
        st.session_state.labels.append(label)
    # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            #this prepends the introductory primer that tells model to be empathetic to first prompt
            #we do this each time since we don't want to display primer in chat history
            l = st.session_state.labels[0]
            p = process_prompt(st.session_state.messages[0]["role"], st.session_state.messages[0]["content"], l)
            primed = [{"role": st.session_state.messages[0]["role"],
                "content": ''.join([primer, '\n', p])}]
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages = primed + [
                    {"role": m["role"], "content": process_prompt(m["role"], m["content"], l)}
                    for m, l in zip(st.session_state.messages[1:], st.session_state.labels[1:])
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.labels.append("-1") #evens out label list. assistants have no emotion tag


if __name__ == "__main__":
    run()
