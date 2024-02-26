# DSC180A-B09-2 website
<!--To create line break: use 2 spaces after a line or use <br>-->
Lohit Geddam lgeddam@ucsd.edu  

Nicholas Shor nshor@ucsd.edu  

Irisa Jin irjin@ucsd.edu  

Henry Luu hluu@ucsd.edu  



B09 Dr. Ali Arsanjani

**Introduction/Background**  

**Dataset**


**Methodology**

**Results**

**Discussion**

**Gradio Chatbox**
```python
import gradio as gr

def greet(name):
    return "Hello, " + name + "!"
iface = gr.Interface(fn=greet, inputs="text", outputs="text", title="Greeting Chatbox")
iface.launch()
```
