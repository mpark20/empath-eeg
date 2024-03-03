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

from urllib.error import URLError

import altair as alt
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code


st.set_page_config(page_title="background")
st.markdown("# Background")
st.write(
"""
Can a computer truly empathize with someone???

Emotion is conveyed in so many ways that simply cannot be recognized through 
the written word alone. Non-verbal cues such as body language, facial 
expression, and gestures can reveal a person's true feelings when words fail. 
However, even these cues can be too subtle for the average human to detect. 

What if we had something even closer? Where intentional cues flake, biometrics 
don't lie. EmpathEEG is a chatbot companion that detects a your emotional state 
via EEG and adjusts its responses to match your energy. 

"""
)
st.image('empatheeg.jpg')
