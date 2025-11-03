import datetime
import os
import re
import time
import nomic
import pandas as pd
import numpy as np
from injector import inject
from nomic import AtlasDataset, atlas
from tenacity import retry, stop_after_attempt, wait_exponential
from ai_ta_backend.database.sql import SQLDatabase
from ai_ta_backend.service.sentry_service import SentryService
from ollama import Client


class NomicService:

    @inject
    def __init__(self, sentry: SentryService, sql: SQLDatabase):
        nomic.cli.login(os.environ['NOMIC_API_KEY'])
        self.ollama_client = Client(host=os.environ['OLLAMA_SERVER_URL'])
        self.sentry = sentry
        self.sql = sql

    # ✅ Helper to fix 'dict' object has no attribute 'data'
    def _safe_data(self, response):
        """Safely extract .data from SQL results or dicts/lists."""
        if response is None:
            return []
        if isinstance(response, dict):
            return response.get("data", [])
        if hasattr(response, "data"):
            return response.data
        if isinstance(response, list):
            return response
        return []

    # ✅ Helper to fix 'dict' object has no attribute 'count'
    def _safe_count(self, response):
        """Safely extract .count from SQL responses (supports dicts, lists, or objects)."""
        if response is None:
            return 0
        if isinstance(response, dict):
            if "count" in response:
                return response["count"]
            if isinstance(response.get("data"), list) and len(response["data"]) > 0:
                if "count" in response["data"][0]:
                    return response["data"][0]["count"]
            return 0
        if hasattr(response, "count"):
            return response.count
        return 0

    # ---------------------------------------------------------------------------------------------
    # Get Nomic Map
    # ---------------------------------------------------------------------------------------------
    def get_nomic_map(self, course_name: str, type: str):
        if not course_name or not type:
            raise ValueError("Course name and type are required")
        if type.lower() not in ['document', 'conversation']:
            raise ValueError("Invalid map type")

        start_time = time.monotonic()
        try:
            field = 'document_map_index' if type.lower() == 'document' else 'conversation_map_index'
            map_name_data = self._safe_data(self.sql.getProjectMapName(course_name, field))

            if not map_name_data or not map_name_data[0].get(field):
                print(f"No {field} found for course: {course_name}")
                return {"map_id": None, "map_link": None}

            map_name = map_name_data[0][field]
            project_name = map_name.split("_index")[0]
            project = AtlasDataset(project_name)
            nomic_map = project.get_map(map_name)

            print(f"⏰ Nomic Full Map Retrieval: {(time.monotonic() - start_time):.2f} seconds")
            return {"map_id": f"iframe{nomic_map.id}", "map_link": nomic_map.map_link}

        except Exception as e:
            print("ERROR in get_nomic_map():", e)
            self.sentry.capture_exception(e)
            return {"map_id": None, "map_link": None}

    # ---------------------------------------------------------------------------------------------
    # Update Conversation Maps
    # ---------------------------------------------------------------------------------------------
    def update_conversation_maps(self):
        try:
            projects = self._safe_data(self.sql.getConvoMapDetails())
            print("Number of projects:", len(projects))

            for project in projects:
                course_name = project['course_name']
                print(f"Processing course: {course_name}")

                if not project.get('convo_map_id') or project['convo_map_id'] == 'N/A':
                    print(f"Creating new conversation map for {course_name}")
                    self.create_conversation_map(course_name)
                    continue

                last_uploaded_id = project['last_uploaded_convo_id']
                count_resp = self.sql.getCountFromLLMConvoMonitor(course_name, last_id=last_uploaded_id)
                total_convo_count = self._safe_count(count_resp)

                if total_convo_count == 0:
                    print("No new conversations to log.")
                    continue

                combined_dfs = []
                current_count = 0
                while current_count < total_convo_count:
                    response_data = self._safe_data(
                        self.sql.getAllConversationsBetweenIds(course_name, last_uploaded_id, 0, 100)
                    )
                    if not response_data:
                        break

                    combined_dfs.append(pd.DataFrame(response_data))
                    current_count += len(response_data)

                    if combined_dfs:
                        final_df = pd.concat(combined_dfs, ignore_index=True)
                        embeddings, metadata = self.data_prep_for_convo_map(final_df)
                        map_name = re.sub(r'[^a-zA-Z0-9\s-]', '',
                                          f"Conversation Map for {course_name}".replace("_", "-")).replace(" ", "-").lower()

                        result = self.append_to_map(embeddings, metadata, map_name)
                        if result == "success":
                            last_uploaded_id = int(final_df['id'].iloc[-1])
                            self.sql.updateProjects(course_name, {'last_uploaded_convo_id': last_uploaded_id})
                        else:
                            print(f"Error updating conversation map: {result}")
                            break
                        combined_dfs = []

                self.create_map_index(course_name, "first_query", "conversation")
                print(f"✔ Processed all conversations for {course_name}")
                print("-" * 70)
                time.sleep(5)

            print("✅ Finished updating all conversation maps.")
            return "success"

        except Exception as e:
            msg = f"Error in updating conversation maps: {e}"
            print(msg)
            self.sentry.capture_exception(e)
            return msg

    # ---------------------------------------------------------------------------------------------
    # Update Document Maps
    # ---------------------------------------------------------------------------------------------
    def update_document_maps(self):
        DOCUMENT_MAP_PREFIX = "Document Map for "
        BATCH_SIZE = 100
        UPLOAD_THRESHOLD = 500

        try:
            projects = self._safe_data(self.sql.getDocMapDetails())
            print("Number of projects:", len(projects))

            for project in projects:
                try:
                    course_name = project['course_name']
                    if not project.get('doc_map_id') or project['doc_map_id'] == 'N/A':
                        print(f"Creating new document map for {course_name}")
                        self.create_document_map(course_name)
                        continue

                    last_uploaded_doc_id = project['last_uploaded_doc_id']
                    response = self.sql.getCountFromDocuments(course_name, last_id=last_uploaded_doc_id)
                    total_doc_count = self._safe_count(response)
                    if total_doc_count == 0:
                        print("No new documents to log.")
                        continue

                    project_name = re.sub(r'[^a-zA-Z0-9\s-]', '',
                                          f"{DOCUMENT_MAP_PREFIX}{course_name}".replace(" ", "-").replace("_", "-").lower())
                    first_id = last_uploaded_doc_id

                    combined_dfs = []
                    current_doc_count = 0
                    while current_doc_count < total_doc_count:
                        response_data = self._safe_data(
                            self.sql.getDocsForIdsGte(course_name, first_id, limit=BATCH_SIZE)
                        )
                        if not response_data:
                            break

                        df = pd.DataFrame(response_data)
                        combined_dfs.append(df)
                        current_doc_count += len(response_data)

                        if current_doc_count >= total_doc_count:
                            final_df = pd.concat(combined_dfs, ignore_index=True)
                            embeddings, metadata = self.data_prep_for_doc_map(final_df)
                            if not embeddings.size:
                                print("No embeddings found. Skipping.")
                                break

                            result = self.append_to_map(embeddings, metadata, project_name)
                            if result == "success":
                                last_id = int(final_df['id'].iloc[-1])
                                self.sql.updateProjects(course_name, {'last_uploaded_doc_id': last_id})
                            else:
                                print(f"Upload failed: {result}")
                                break

                            combined_dfs = []

                    self.create_map_index(course_name, "text", "document")
                    print(f"✔ Completed document updates for {course_name}")
                    print("-" * 70)
                    time.sleep(5)

                except Exception as e:
                    print(f"Error updating document map for {course_name}: {e}")
                    self.sentry.capture_exception(e)
                    continue

            return "success"

        except Exception as e:
            print(f"Error in update_document_maps: {e}")
            self.sentry.capture_exception(e)
            return f"Error in update_document_maps: {e}"

    # ---------------------------------------------------------------------------------------------
    # Create Conversation Map
    # ---------------------------------------------------------------------------------------------
    def create_conversation_map(self, course_name: str):
        NOMIC_MAP_NAME_PREFIX = 'Conversation Map for '
        BATCH_SIZE = 100
        MIN_CONVERSATIONS = 20
        UPLOAD_THRESHOLD = 500

        try:
            existing_map = self._safe_data(self.sql.getConvoMapFromProjects(course_name))
            if existing_map and existing_map[0].get('convo_map_id'):
                return "Map already exists for this course."

            response = self.sql.getCountFromLLMConvoMonitor(course_name, last_id=0)
            total_convo_count = self._safe_count(response)
            if total_convo_count < MIN_CONVERSATIONS:
                return "Cannot create map: not enough conversations."

            project_name = re.sub(r'[^a-zA-Z0-9\s-]', '',
                                  (NOMIC_MAP_NAME_PREFIX + course_name).replace(" ", "-").replace("_", "-").lower())
            first_id = 0

            combined_dfs = []
            current_convo_count = 0
            first_batch = True

            while current_convo_count < total_convo_count:
                response_data = self._safe_data(
                    self.sql.getAllConversationsBetweenIds(course_name, first_id, 0, BATCH_SIZE)
                )
                if not response_data:
                    break

                df = pd.DataFrame(response_data)
                combined_dfs.append(df)
                current_convo_count += len(response_data)

                if current_convo_count >= total_convo_count or len(combined_dfs) >= UPLOAD_THRESHOLD:
                    final_df = pd.concat(combined_dfs, ignore_index=True)
                    embeddings, metadata = self.data_prep_for_convo_map(final_df)
                    if not embeddings.size:
                        break

                    if first_batch:
                        result = self.create_map(embeddings, metadata,
                                                 f"{NOMIC_MAP_NAME_PREFIX}{course_name}",
                                                 f"{course_name}_convo_index",
                                                 "first_query")
                        first_batch = False
                    else:
                        result = self.append_to_map(embeddings, metadata, project_name)

                    if result == "success":
                        project = AtlasDataset(project_name)
                        last_id = int(final_df['id'].iloc[-1])
                        self.sql.insertProject({
                            'course_name': course_name,
                            'convo_map_id': project.id,
                            'last_uploaded_convo_id': last_id
                        })
                    else:
                        print(f"Error creating conversation map: {result}")
                        return result

                    combined_dfs = []

            self.create_map_index(course_name, "first_query", "conversation")
            return "success"

        except Exception as e:
            print(f"Error in create_conversation_map: {e}")
            self.sentry.capture_exception(e)
            return f"Error in create_conversation_map: {e}"

    # ---------------------------------------------------------------------------------------------
    # Create Document Map
    # ---------------------------------------------------------------------------------------------
    def create_document_map(self, course_name: str):
        DOCUMENT_MAP_PREFIX = "Document Map for "
        BATCH_SIZE = 100
        MIN_DOCUMENTS = 20
        UPLOAD_THRESHOLD = 500

        try:
            existing_map = self._safe_data(self.sql.getDocMapFromProjects(course_name))
            if existing_map and existing_map[0].get('doc_map_id'):
                return "Map already exists."

            response = self.sql.getCountFromDocuments(course_name, last_id=0)
            total_doc_count = self._safe_count(response)
            if total_doc_count < MIN_DOCUMENTS:
                return "Cannot create map: not enough documents."

            project_name = re.sub(r'[^a-zA-Z0-9\s-]', '',
                                  (DOCUMENT_MAP_PREFIX + course_name).replace(" ", "-").replace("_", "-").lower())
            first_id = 0
            combined_dfs = []
            first_batch = True
            current_doc_count = 0

            while current_doc_count < total_doc_count:
                response_data = self._safe_data(
                    self.sql.getDocsForIdsGte(course_name, first_id, limit=BATCH_SIZE)
                )
                if not response_data:
                    break

                df = pd.DataFrame(response_data)
                combined_dfs.append(df)
                current_doc_count += len(response_data)

                if current_doc_count >= total_doc_count or len(combined_dfs) >= UPLOAD_THRESHOLD:
                    final_df = pd.concat(combined_dfs, ignore_index=True)
                    embeddings, metadata = self.data_prep_for_doc_map(final_df)
                    if not embeddings.size:
                        break

                    if first_batch:
                        result = self.create_map(embeddings, metadata,
                                                 f"{DOCUMENT_MAP_PREFIX}{course_name}",
                                                 f"{course_name}_doc_index",
                                                 "text")
                        first_batch = False
                    else:
                        result = self.append_to_map(embeddings, metadata, project_name)

                    if result == "success":
                        project = AtlasDataset(project_name)
                        last_id = int(final_df['id'].iloc[-1])
                        self.sql.insertProject({
                            'course_name': course_name,
                            'doc_map_id': project.id,
                            'last_uploaded_doc_id': last_id
                        })
                    else:
                        print(f"Error creating document map: {result}")
                        return result

                    combined_dfs = []

            self.create_map_index(course_name, "text", "document")
            return "success"

        except Exception as e:
            print(f"Error in create_document_map: {e}")
            self.sentry.capture_exception(e)
            return f"Error in create_document_map: {e}"

    def clean_up_conversation_maps(self):
      """
      Deletes all Nomic maps and re-creates them. To be called weekly via a CRON job.
      This is to clean up all the new map indices generated daily.
      """
      try:
        # step 1: get all conversation maps from SQL
        data = self.sql.getProjectsWithConvoMaps().data
        print("Length of projects: ", len(data))
        # step 2: delete all conversation maps from Nomic
        for project in data:
          try:
            project_name = re.sub(r'[^a-zA-Z0-9\s-]', '',
                                  (f"Conversation Map for {project['course_name']}").replace(" ",
                                                                                            "-").replace("_",
                                                                                                          "-").lower())
            print(f"Deleting conversation map: {project_name}")
            dataset = AtlasDataset(project_name)
            dataset.delete()

            # step 3: update SQL table to remove map info
            self.sql.updateProjects(project['course_name'], {
                'convo_map_id': None,
                'last_uploaded_convo_id': None,
                'conversation_map_index': None
            })

          except Exception as e:
            print(f"Error in deleting conversation map: {e}")
            self.sentry.capture_exception(e)
            continue
        print("Deleted all conversation maps.")

        # step 4: re-create conversation maps by calling update function
        status = self.update_conversation_maps()  # this function will create new maps if not already present!
        print("Map re-creation status: ", status)

        return "success"
      except Exception as e:
        print(e)
        self.sentry.capture_exception(e)
        return f"Error in cleaning up conversation maps: {str(e)}"

    def clean_up_document_maps(self):
      """
      Deletes all Nomic maps and re-creates them. To be called weekly via a CRON job.
      This is to clean up all the new map indices generated daily.
      """
      try:
        # step 1: get all document maps from SQL
        data = self.sql.getProjectsWithDocMaps().data
        print("Length of projects: ", len(data))
        # step 2: delete all document maps from Nomic
        for project in data:
          try:
            project_name = re.sub(r'[^a-zA-Z0-9\s-]', '',
                                  (f"Document Map for {project['course_name']}").replace(" ", "-").replace("_",
                                                                                                          "-").lower())
            print(f"Deleting document map: {project_name}")
            dataset = AtlasDataset(project_name)
            dataset.delete()

            # step 3: update SQL table to remove map info
            self.sql.updateProjects(project['course_name'], {
                'doc_map_id': None,
                'last_uploaded_doc_id': None,
                'document_map_index': None
            })

          except Exception as e:
            print(f"Error in deleting document map: {e}")
            self.sentry.capture_exception(e)
            continue

        # step 4: re-create conversation maps by calling update function
        status = self.update_document_maps()  # this function will create new maps if not already present!
        print("Map re-creation status: ", status)

        return "success"
      except Exception as e:
        print(e)
        self.sentry.capture_exception(e)
        return f"Error in cleaning up document maps: {str(e)}"


  #   ## -------------------------------- SUPPLEMENTARY MAP FUNCTIONS --------------------------------- ##

    def rebuild_map(self, course_name: str, map_type: str):
      """
      Rebuilds a given map in Nomic.
      Args:
          course_name (str): Name of the course
          map_type (str): Type of map ('document' or 'conversation')
      Returns:
          str: Status of map rebuilding process
      """
      MAP_PREFIXES = {'document': 'Document Map for ', 'conversation': 'Conversation Map for '}

      try:
        project_name = re.sub(r'[^a-zA-Z0-9\s-]', '',
                              (MAP_PREFIXES.get(map_type.lower(), '') + course_name).replace(" ",
                                                                                            "-").replace("_",
                                                                                                          "-").lower())
        print(f"Rebuilding map: {project_name}")
        project = AtlasDataset(project_name)

        if project.is_accepting_data:
          project.update_indices(rebuild_topic_models=True)

        return "success"

      except Exception as e:
        print(e)
        self.sentry.capture_exception(e)
        return f"Error in rebuilding map: {e}"

    def create_map_index(self, course_name: str, index_field: str, map_type: str):
      """
      Creates a new index for a given map in Nomic.
      """
      MAP_PREFIXES = {'document': 'Document Map for ', 'conversation': 'Conversation Map for '}
      try:
        map_type = map_type.lower()
        # create index
        project_name = re.sub(r'[^a-zA-Z0-9\s-]', '',
                              (MAP_PREFIXES.get(map_type, '') + course_name).replace(" ", "-").replace("_", "-").lower())
        print(f"Creating index for map: {project_name}")

        project = AtlasDataset(project_name)

        #current_day = datetime.datetime.now().day
        index_name = f"{project_name}_index_{datetime.datetime.now().strftime('%Y-%m-%d')}"

        project.create_index(name=index_name, indexed_field=index_field, topic_model=True, duplicate_detection=True)

        # update index name to SQL database
        self.sql.updateProjects(course_name, {f"{map_type}_map_index": index_name})

        return "success"
      except Exception as e:
        print(e)
        self.sentry.capture_exception(e)
        return f"Error in creating index: {e}"

    def create_map(self, embeddings, metadata, map_name, index_name, index_field):
      """
      Creates a Nomic map with topic modeling and duplicate detection.
      
      Args:
          embeddings (np.ndarray): Document embeddings if available
          metadata (pd.DataFrame): Metadata for the map
          map_name (str): Name of the map to create
          index_name (str): Name of the index to create
          index_field (str): Field to be indexed
          
      Returns:
          str: 'success' or error message
      """
      print(f"Creating map: {map_name}")

      try:
          project = AtlasDataset(
              map_name,
              unique_id_field="id",
          )
          
          # Check if embeddings is a non-empty numpy array
          if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
              project.add_data(data=metadata, embeddings=embeddings)
          return "success"

      except Exception as e:
          print(e)
          return f"Error in creating map: {e}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=10, max=600))
    def append_to_map(self, embeddings, metadata, map_name):
      """
      Appends new data to an existing Nomic map.
      
      Args:
          metadata (pd.DataFrame): Metadata for the map update
          map_name (str): Name of the target map
          
      Returns:
          str: 'success' or error message
      """
      try:
          print(f"Appending to map: {map_name}")
          project = AtlasDataset(map_name)

          start_time = time.monotonic()
          while time.monotonic() - start_time < 60:
              if project.is_accepting_data:
                  if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                      project.add_data(data=metadata, embeddings=embeddings)
                  return "success"
              
              print("Project is currently indexing. Waiting for 10 seconds...")
              time.sleep(10)

          return "Project busy"

      except Exception as e:
          print(e)
          return f"Error in appending to map: {e}"

    def data_prep_for_convo_map(self, df: pd.DataFrame) -> list:
      """
      Prepares conversation data from Supabase for Nomic map upload.
      Args:
          df (pd.DataFrame): Dataframe of documents from Supabase   
      Returns:
          pd.DataFrame: Processed metadata for map creation, or None if error occurs
      """
      print("Preparing conversation data for map")

      try:
        metadata = []
        raw_text = []
        current_time = datetime.datetime.now()

        for _, row in df.iterrows():
          created_at = datetime.datetime.strptime(row['created_at'],
                                                  "%Y-%m-%dT%H:%M:%S.%f%z")
          messages = row['convo']['messages']
          first_message = messages[0]['content']
          if isinstance(first_message, list):
            first_message = first_message[0].get('text', '')

          conversation = []
          for message in messages:
            emoji = "🙋 " if message['role'] == 'user' else "🤖 "
            content = message['content']
            text = content[0].get('text', '') if isinstance(content, list) else content
            conversation.append(f"\n>>> {emoji}{message['role']}: {text}\n")

          metadata.append({
              "course": row['course_name'],
              "conversation": ''.join(conversation),
              "conversation_id": row['convo']['id'],
              "id": row['id'],
              "user_email": row['user_email'] or "",
              "first_query": first_message,
              "created_at": created_at,
              "modified_at": current_time
          })
          raw_text.append(first_message)
        
        # generate embeddings using ollama
        response = self.ollama_client.embed(model='nomic-embed-text:v1.5', input=raw_text)
        
        embeddings = response['embeddings']
        embeddings = np.array(embeddings)
        print("Shape of embeddings: ", embeddings.shape)
        
        result = pd.DataFrame(metadata)
        print(f"Metadata shape: {result.shape}")
        return [embeddings, result]

      except Exception as e:
        print(f"Error in data preparation: {e}")
        self.sentry.capture_exception(e)
        return [np.array([]), pd.DataFrame()]

    def data_prep_for_doc_map(self, df: pd.DataFrame) -> list:
      try:
          metadata = []
          embeddings = []
          current_time = datetime.datetime.now()

          for _, row in df.iterrows():
              created_at = datetime.datetime.strptime(row['created_at'], 
                                                    "%Y-%m-%dT%H:%M:%S.%f%z")

              for idx, context in enumerate(row['contexts'], 1):
                  # Validate embedding before adding
                  embedding = context.get('embedding')
                  if embedding is not None and isinstance(embedding, (list, np.ndarray)):
                      # Convert to list if numpy array
                      if isinstance(embedding, np.ndarray):
                          embedding = embedding.tolist()
                      
                      # Check if embedding has the expected dimension
                      if len(embedding) > 0:  # Add your expected dimension check here
                          embeddings.append(embedding)
                          metadata.append({
                              "id": f"{row['id']}_{idx}",
                              "created_at": created_at,
                              "s3_path": row['s3_path'],
                              "url": row['url'] or "",
                              "base_url": row['base_url'] or "",
                              "readable_filename": row['readable_filename'],
                              "modified_at": current_time,
                              "text": context['text']
                          })

          # Convert to numpy array only if we have valid embeddings
          if embeddings and len(embeddings) > 20:
              embeddings = np.array(embeddings)
              print(f"Embeddings shape: {embeddings.shape}")
              return [embeddings, pd.DataFrame(metadata)]
          else:
              print("No valid embeddings found")
              return [np.array([]), pd.DataFrame()]

      except Exception as e:
          print(f"Error in document data preparation: {e}")
          self.sentry.capture_exception(e)
          return [np.array([]), pd.DataFrame()]
