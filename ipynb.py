#!/usr/bin/env python
# coding: utf-8

# # Object Detection with SSD
# ### Here we demostrate detection on example images using SSD with PyTorch

# In[192]:


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
from pascal_voc_writer import Writer
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd


# In[193]:


import json
from graphqlclient import GraphQLClient
client = GraphQLClient('https://api.labelbox.com/graphql')
client.inject_token('Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjanVvMG9kOHUzMmV4MDgxNzRxMW8za2ZtIiwib3JnYW5pemF0aW9uSWQiOiJjanVvMG9kOGM0MmVpMDg3MXc1bWJvcjUxIiwiYXBpS2V5SWQiOiJjanVwZGkyOWt3N2hzMDg3MTZ3dTh6ZW1kIiwiaWF0IjoxNTU1NzU2OTIwLCJleHAiOjIxODY5MDg5MjB9._m24Q4hVQKIJZVi61-y7bqw5iX1he37zSca1VmW027g')

def me():
    res_str = client.execute("""
    query GetUserInformation {
      user {
        id
        organization{
          id
        }
      }
    }
    """)

    res = json.loads(res_str)
    return res['data']['user']


def createDataset(name):
    res_str = client.execute("""
    mutation CreateDatasetFromAPI($name: String!) {
      createDataset(data:{
        name: $name
      }){
        id
      }
    }
    """, {'name': name})

    res = json.loads(res_str)
    return res['data']['createDataset']['id']


def createProject(name):
    res_str = client.execute("""
    mutation CreateProjectFromAPI($name: String!) {
      createProject(data:{
        name: $name
      }){
        id
      }
    }
    """, {'name': name})

    res = json.loads(res_str)
    return res['data']['createProject']['id']


def completeSetupOfProject(project_id, dataset_id, labeling_frontend_id):
    res_str = client.execute("""
    mutation CompleteSetupOfProject($projectId: ID!, $datasetId: ID!, $labelingFrontendId: ID!){
      updateProject(
        where:{
          id:$projectId
        },
        data:{
          setupComplete: "2018-11-29T20:46:59.521Z",
          datasets:{
            connect:{
              id:$datasetId
            }
          },
          labelingFrontend:{
            connect:{
              id:$labelingFrontendId
            }
          }
        }
      ){
        id
      }
    }
    """, {
        'projectId': project_id,
        'datasetId': dataset_id,
        'labelingFrontendId': labeling_frontend_id
    })

    res = json.loads(res_str)
    return res['data']['updateProject']['id']


def configure_interface_for_project(ontology, project_id, interface_id, organization_id):
    res_str = client.execute("""
      mutation ConfigureInterfaceFromAPI($projectId: ID!, $customizationOptions: String!, $labelingFrontendId: ID!, $organizationId: ID!) {
        createLabelingFrontendOptions(data:{
          customizationOptions: $customizationOptions,
          project:{
            connect:{
              id: $projectId
            }
          }
          labelingFrontend:{
            connect:{
              id:$labelingFrontendId
            }
          }
          organization:{
            connect:{
              id: $organizationId
            }
          }
        }){
          id
        }
      }
    """, {
        'projectId': project_id,
        'customizationOptions': json.dumps(ontology),
        'labelingFrontendId': interface_id,
        'organizationId': organization_id,
    })

    res = json.loads(res_str)
    return res['data']['createLabelingFrontendOptions']['id']


def get_image_labeling_interface_id():
    res_str = client.execute("""
      query GetImageLabelingInterfaceId {
        labelingFrontends(where:{
          iframeUrlPath:"https://image-segmentation-v4.labelbox.com"
        }){
          id
        }
      }
    """)

    res = json.loads(res_str)
    return res['data']['labelingFrontends'][0]['id']


def create_prediction_model(name, version):
    res_str = client.execute("""
      mutation CreatePredictionModelFromAPI($name: String!, $version: Int!) {
        createPredictionModel(data:{
          name: $name,
          version: $version
        }){
          id
        }
      }
    """, {
      'name': name,
      'version': version
    })

    res = json.loads(res_str)
    return res['data']['createPredictionModel']['id']

def attach_prediction_model_to_project(prediction_model_id, project_id):
    res_str = client.execute("""
      mutation AttachPredictionModel($predictionModelId: ID!, $projectId: ID!){
        updateProject(where:{
          id: $projectId
        }, data:{
          activePredictionModel:{
            connect:{
              id: $predictionModelId
            }
          }
        }){
          id
        }
      }
    """, {
      'predictionModelId': prediction_model_id,
      'projectId': project_id
    })

    res = json.loads(res_str)
    return res['data']['updateProject']['id']


# Make sure you pass in label as string
# json.dumps(label_json)
def create_label(label, project_id, data_row_id):
    res_str = client.execute("""
    mutation CreateLabelFromApi($label: String!, $projectId: ID!, $dataRowId: ID!){
      createLabel(data:{
        label:$label,
        secondsToLabel:0,
        project:{
          connect:{
            id:$projectId
          }
        }
        dataRow:{
          connect:{
            id:$dataRowId
          }
        }
        type:{
          connect:{
            name:"Any"
          }
        }
      }){
      id
      }
    }
    """, {
        'label': label,
        'projectId': project_id,
        'dataRowId': data_row_id
    })

    res = json.loads(res_str)
    return res['data']['createLabel']['id']

def create_prediction(label, prediction_model_id, project_id, data_row_id):
    res_str = client.execute("""
      mutation CreatePredictionFromAPI($label: String!, $predictionModelId: ID!, $projectId: ID!, $dataRowId: ID!) {
        createPrediction(data:{
          label: $label,
          predictionModelId: $predictionModelId,
          projectId: $projectId,
          dataRowId: $dataRowId,
        }){
          id
        }
      }
    """, {
        'label': label,
        'predictionModelId': prediction_model_id,
        'projectId': project_id,
        'dataRowId': data_row_id
    })

    res = json.loads(res_str)
    return res['data']['createPrediction']['id']


def create_datarow(row_data, external_id, dataset_id):
    res_str = client.execute("""
      mutation CreateDataRowFromAPI(
        $rowData: String!,
        $externalId: String,
        $datasetId: ID!
      ) {
        createDataRow(data:{
          externalId: $externalId,
          rowData: $rowData,
          dataset:{
            connect:{
              id: $datasetId
            }
          }
        }){
          id
        }
      }
    """, {
        'rowData': row_data,
        'externalId': external_id,
        'datasetId': dataset_id
    })

    res = json.loads(res_str)
    return res['data']['createDataRow']['id']


# ## Build SSD300 in Test Phase
# 1. Build the architecture, specifyingsize of the input image (300),
#     and number of object classes to score (21 for VOC dataset)
# 2. Next we load pretrained weights on the VOC0712 trainval dataset  

# In[194]:


net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('../weights/ssd300_COCO_20000.pth')


# ## Load Image 
# ### Here we just load a sample image from the VOC07 dataset 

# In[220]:


def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success :
            if count % 100 == 0:
                cv2.imwrite(os.path.join(path_output_dir, '%d.jpeg') % count, image)
                count += 1
            count+=1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
#video_to_frames("C:/diploma/video.avi","C:/diploma/frames/")
#source = 'C:/diploma/ssd-pytorch/frames/'
#for root, dirs, filenames in os.walk(source):
#for f in filenames:
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('../weights/ssd300_COCO_20000.pth')
for image_number in [23400,16600,15100,15400,16900,23200,23100,22900,11400,11300,700,600,2500,8300,10500]:
    image = cv2.imread('C:/diploma/ssd-pytorch/frames/{}.jpeg'.format(image_number), cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
    height, width, channels = image.shape
    #writer = Writer('karpin.jpg',width,height)
    get_ipython().run_line_magic('matplotlib', 'inline')
    from matplotlib import pyplot as plt
    from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
    # here we specify year (07 or 12) and dataset ('test', 'val', 'train') 
    #testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
    #img_id = 97
    #image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    plt.figure(figsize=(10,10))
    plt.imshow(rgb_image)
    plt.show()
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    #x = x[:, :, ::-1].copy()
    plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    from data import VOC_CLASSES as labels
    top_k=10

    plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    data = [{
                "prediction_label": {
                  "person": [
                  ]
                },
                "image_url": "http://192.168.1.48:8000/{}.jpeg".format(image_number),
                "external_id": "local_image{}".format(image_number)
              }
            ]
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            #print(detections[0,i,j,0])
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            data[0]['prediction_label']['person'].append({'geometry':[{"x":int(pt[0]),"y":int(pt[1]) },
                                    {"x": int(pt[2]), "y": int(pt[3])}]})
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1
    image_number += 100
    user_info = me()
    org_id = user_info['organization']['id']

    project_id = "cjvak22atnklk0800eifi2yct"
    dataset_id = "cjvaluw3lntxh08006u2jkkjw"

    interface_id = get_image_labeling_interface_id()
    ontology = {
        "tools": [
            {
                  "color": "navy",
                  "tool": "regtangle",
                  "name": "Football"
            }
          ]
      }

    configure_interface_for_project(
    ontology, project_id, interface_id, org_id)
    completeSetupOfProject(project_id, dataset_id, interface_id)
    print('Attached Dataset and Interface to Created Project')

    prediction_model_id = create_prediction_model('Ollie Example Model', 1)
    attach_prediction_model_to_project(prediction_model_id, project_id)

    print('Created and attached prediction model: %s' % (prediction_model_id))

    for row in data:
        data_row_id = create_datarow(row['image_url'], row['external_id'], dataset_id)
        print('Created DataRow: %s' % (data_row_id))

    #    prediction_id = create_prediction(json.dumps(row['prediction_label']), prediction_model_id, project_id, data_row_id)
        prediction_id = create_label(json.dumps(row['prediction_label']), project_id, data_row_id)
        print('Created Prediction: %s %s' , (prediction_id), (data_row_id))


# ## Pre-process the input.  
# #### Using the torchvision package, we can create a Compose of multiple built-in transorm ops to apply 
# For SSD, at test time we use a custom BaseTransform callable to
# resize our image to 300x300, subtract the dataset's mean rgb values, 
# and swap the color channels for input to SSD300.

# In[210]:


x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
#x = x[:, :, ::-1].copy()
plt.imshow(x)
x = torch.from_numpy(x).permute(2, 0, 1)


# ## SSD Forward Pass
# ### Now just wrap the image in a Variable so it is recognized by PyTorch autograd

# In[211]:


xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)


# ## Parse the Detections and View Results
# Filter outputs with confidence scores lower than a threshold 
# Here we choose 60% 

# In[215]:


from data import VOC_CLASSES as labels
top_k=10

plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
data = [{
            "prediction_label": {
              "person": [
              ]
            },
            "image_url": "http://192.168.1.48:8000/{}.jpeg".format(image_number),
            "external_id": "local_image"
          }
        ]
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        #print(detections[0,i,j,0])
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        data[0]['prediction_label']['person'].append({'geometry':[{"x":int(pt[0]),"y":int(pt[1]) },
                                {"x": int(pt[2]), "y": int(pt[3])}]})
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1

user_info = me()
org_id = user_info['organization']['id']

project_id = "cjvak22atnklk0800eifi2yct"
dataset_id = "cjvaluw3lntxh08006u2jkkjw"

interface_id = get_image_labeling_interface_id()
ontology = {
    "tools": [
        {
              "color": "navy",
              "tool": "regtangle",
              "name": "Football"
        }
      ]
  }

configure_interface_for_project(
ontology, project_id, interface_id, org_id)
completeSetupOfProject(project_id, dataset_id, interface_id)
print('Attached Dataset and Interface to Created Project')

prediction_model_id = create_prediction_model('Ollie Example Model', 1)
attach_prediction_model_to_project(prediction_model_id, project_id)

print('Created and attached prediction model: %s' % (prediction_model_id))

for row in data:
    data_row_id = create_datarow(row['image_url'], row['external_id'], dataset_id)
    print('Created DataRow: %s' % (data_row_id))

#    prediction_id = create_prediction(json.dumps(row['prediction_label']), prediction_model_id, project_id, data_row_id)
    prediction_id = create_label(json.dumps(row['prediction_label']), project_id, data_row_id)
    print('Created Prediction: %s %s' , (prediction_id), (data_row_id))
#writer.addObject('label_name', int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]))        
#writer.save('golovin.xml')


# In[165]:


for i in range(1,3):
    print(i)
    


# In[ ]:





# In[ ]:




