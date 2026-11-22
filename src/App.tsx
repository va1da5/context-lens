import { useEffect, useMemo, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

type ContextContentPart = {
  text?: string
  refusal?: string
  type?: string
  [key: string]: unknown
}

type ContextEntry = {
  role?: string | null
  type?: string | null
  name?: string | null
  content?: string | ContextContentPart[]
  arguments?: unknown
  output?: unknown
  call_id?: string | null
  status?: string | null
  id?: string | null
  [key: string]: unknown
}

type ContextTool = {
  type?: string
  name: string
  description?: string
  [key: string]: unknown
}

type ContextDocument = {
  model?: string
  instructions?: string | ContextEntry[]
  input: ContextEntry[]
  tools: ContextTool[]
  source: Record<string, unknown>
}

type Persona = {
  id: string
  label: string
  description: string
  color: string
}

const personaDetails: Record<string, Omit<Persona, 'id'>> = {
  system: { label: 'System', description: 'System instructions', color: 'system' },
  developer: { label: 'Developer', description: 'Developer instructions', color: 'developer' },
  user: { label: 'User', description: 'Requests & input', color: 'user' },
  assistant: { label: 'Assistant', description: 'Visible responses', color: 'assistant' },
  reasoning: { label: 'Reasoning', description: 'Internal thought', color: 'reasoning' },
  function_call: { label: 'Tool calls', description: 'Actions requested', color: 'tool' },
  function_call_output: { label: 'Tool results', description: 'Returned outputs', color: 'result' },
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseEntries(value: unknown, field: string): ContextEntry[] {
  if (!Array.isArray(value)) throw new Error(`The "${field}" field must be an array.`)

  return value.map((entry, index) => {
    if (!isRecord(entry)) throw new Error(`${field} item ${index + 1} must be a JSON object.`)
    if (entry.content !== undefined && typeof entry.content !== 'string' && !Array.isArray(entry.content)) {
      throw new Error(`${field} item ${index + 1} has an unsupported "content" value.`)
    }
    if (Array.isArray(entry.content) && entry.content.some((part) => !isRecord(part))) {
      throw new Error(`${field} item ${index + 1} has an invalid content part.`)
    }
    for (const key of ['role', 'type', 'name', 'status']) {
      if (entry[key] != null && typeof entry[key] !== 'string') {
        throw new Error(`${field} item ${index + 1} has an invalid "${key}" field.`)
      }
    }
    return entry as ContextEntry
  })
}

function parseContext(value: unknown): ContextDocument {
  if (!isRecord(value)) throw new Error('The JSON must contain a context object.')
  if (value.model !== undefined && typeof value.model !== 'string') {
    throw new Error('The "model" field must be a string.')
  }
  if (value.instructions != null && typeof value.instructions !== 'string' && !Array.isArray(value.instructions)) {
    throw new Error('The "instructions" field must be a string or an array of messages.')
  }

  const rawInput = value.input ?? []
  if (typeof rawInput !== 'string' && !Array.isArray(rawInput)) {
    throw new Error('The "input" field must be a string or an array of items.')
  }
  const input = typeof rawInput === 'string'
    ? [{ role: 'user', content: rawInput }]
    : parseEntries(rawInput, 'input')
  const output = value.output == null ? [] : parseEntries(value.output, 'output')
  const instructions = typeof value.instructions === 'string'
    ? value.instructions
    : Array.isArray(value.instructions)
      ? parseEntries(value.instructions, 'instructions')
      : undefined

  const rawTools = value.tools ?? []
  if (!Array.isArray(rawTools)) throw new Error('The "tools" field must be an array.')
  const tools = rawTools.map((tool, index) => {
    if (!isRecord(tool)) {
      throw new Error(`Tool ${index + 1} must be a JSON object.`)
    }
    if (tool.type != null && typeof tool.type !== 'string') {
      throw new Error(`Tool ${index + 1} has an invalid "type".`)
    }
    if (tool.description != null && typeof tool.description !== 'string') {
      throw new Error(`Tool ${index + 1} has an invalid "description".`)
    }
    const name = typeof tool.name === 'string' ? tool.name : tool.type
    if (typeof name !== 'string') {
      throw new Error(`Tool ${index + 1} must have a string "name" or "type".`)
    }
    return {
      ...tool,
      type: typeof tool.type === 'string' ? tool.type : undefined,
      name,
      description: typeof tool.description === 'string' ? tool.description : undefined,
    }
  })

  return {
    input: [...input, ...output],
    tools,
    source: value,
    ...(typeof value.model === 'string' ? { model: value.model } : {}),
    ...(instructions !== undefined ? { instructions } : {}),
  }
}

const initialContext: ContextDocument = { input: [], tools: [], source: {} }

function formatPersona(id: string): Persona {
  return {
    id,
    ...(personaDetails[id] ?? {
      label: id.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()),
      description: 'Context messages',
      color: 'unknown',
    }),
  }
}

function pretty(value: unknown): string {
  if (typeof value !== 'string') return JSON.stringify(value, null, 2) ?? ''
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}

function JsonHighlight({ text }: { text: string }) {
  const tokenPattern = /("(?:\\.|[^"\\])*")(\s*:)?|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b|\b(?:true|false|null)\b|[\u007B\u007D\u005B\u005D\u002C\u003A]/g
  const parts: React.ReactNode[] = []
  let previousIndex = 0

  for (const match of text.matchAll(tokenPattern)) {
    const index = match.index ?? 0
    if (index > previousIndex) parts.push(text.slice(previousIndex, index))

    if (match[1]) {
      parts.push(<span className="json-key" key={`${index}-key`}>{match[1]}</span>)
      if (match[2]) parts.push(<span className="json-punctuation" key={`${index}-colon`}>{match[2]}</span>)
      previousIndex = index + match[0].length
      continue
    }

    const token = match[0]
    let className = 'json-punctuation'
    if (token.startsWith('"')) className = match[2] ? 'json-key' : 'json-string'
    else if (token === 'true' || token === 'false') className = 'json-boolean'
    else if (token === 'null') className = 'json-null'
    else if (/^-?\d/.test(token)) className = 'json-number'

    parts.push(<span className={className} key={`${index}-${token}`}>{token}</span>)
    previousIndex = index + token.length
  }
  if (previousIndex < text.length) parts.push(text.slice(previousIndex))

  return <>{parts}</>
}

function contentPartText(value: unknown): string {
  if (!isRecord(value)) return pretty(value)
  if (typeof value.text === 'string') return value.text
  if (typeof value.refusal === 'string') return value.refusal
  if (typeof value.type === 'string') {
    return `[${formatPersona(value.type).label} content; see raw API item for details]`
  }
  return pretty(value)
}

function entryText(entry: ContextEntry) {
  if (typeof entry.content === 'string') return entry.content
  if (Array.isArray(entry.content)) {
    return entry.content.map(contentPartText).join('\n\n')
  }
  if (Array.isArray(entry.summary)) {
    return entry.summary.map(contentPartText).join('\n\n')
  }
  if (entry.arguments !== undefined) return pretty(entry.arguments)
  if (entry.output !== undefined) {
    return Array.isArray(entry.output)
      ? entry.output.map(contentPartText).join('\n\n')
      : pretty(entry.output)
  }
  return pretty(entry)
}

function instructionsText(instructions: ContextDocument['instructions']): string {
  if (typeof instructions === 'string') return instructions
  if (!instructions) return ''
  return instructions.map((entry) => {
    const role = entry.role ?? entry.type ?? 'Instruction'
    return `**${formatPersona(role).label}**\n\n${entryText(entry)}`
  }).join('\n\n---\n\n')
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
          img: ({ src, alt }) => <img src={src} alt={alt ?? ''} loading="lazy" referrerPolicy="no-referrer" />,
          code: ({ className, children, ...props }) => className?.includes('language-json')
            ? <code className={className} {...props}><JsonHighlight text={String(children).replace(/\n$/, '')} /></code>
            : <code className={className} {...props}>{children}</code>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

function Icon({ name, size = 16 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5M3 16l9 5 9-5" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    chevron: <path d="m9 18 6-6-6-6" />,
    code: <><path d="m8 17-5-5 5-5m8 10 5-5-5-5m-2-3-4 16" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    panel: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M15 4v16" /></>,
    close: <><path d="m18 6-12 12M6 6l12 12" /></>,
    down: <path d="m7 10 5 5 5-5" />,
  }
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      {paths[name] ?? paths.layers}
    </svg>
  )
}

function App() {
  const [contextData, setContextData] = useState(initialContext)
  const [filename, setFilename] = useState('context.json')
  const [fileError, setFileError] = useState('')
  const [showImport, setShowImport] = useState(false)
  const [pastedJson, setPastedJson] = useState('')
  const [activePersona, setActivePersona] = useState('all')
  const [search, setSearch] = useState('')
  const [hideReasoning, setHideReasoning] = useState(false)
  const searchInput = useRef<HTMLInputElement>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const [showTools, setShowTools] = useState(false)

  const entries = useMemo(() => contextData.input.map((entry, index) => ({
    ...entry,
    index,
    persona: entry.role ?? entry.type ?? 'unknown',
  })), [contextData])
  const personas = useMemo(() => {
    const ids = [...new Set(entries.map((entry) => entry.persona))]
    return ids.map(formatPersona)
  }, [entries])
  const tools = contextData.tools
  const instructions = instructionsText(contextData.instructions)
  const hasContext = entries.length > 0 || Boolean(instructions) || tools.length > 0 || Boolean(contextData.model)
  const title = 'Context messages'

  const personaCounts = useMemo(
    () => Object.fromEntries(personas.map((persona) => [persona.id, entries.filter((entry) => entry.persona === persona.id).length])),
    [entries, personas],
  )
  const filteredEntries = useMemo(() => {
    const query = search.trim().toLowerCase()
    return entries.filter((entry) => {
      if (activePersona !== 'all' && entry.persona !== activePersona) return false
      if (hideReasoning && entry.persona === 'reasoning') return false
      if (query && !`${entry.persona} ${entry.name ?? ''} ${entryText(entry)}`.toLowerCase().includes(query)) return false
      return true
    })
  }, [activePersona, entries, hideReasoning, search])

  function loadContextJson(json: string, sourceName: string) {
    try {
      const parsed = parseContext(JSON.parse(json) as unknown)
      setContextData(parsed)
      setFilename(sourceName)
      setFileError('')
      setActivePersona('all')
      setSearch('')
      setHideReasoning(false)
      setShowImport(false)
    } catch (error) {
      setFileError(error instanceof Error ? error.message : 'Unable to load this JSON.')
    }
  }

  async function loadContextFile(file: File | undefined) {
    if (!file) return
    try {
      loadContextJson(await file.text(), file.name)
    } catch (error) {
      setFileError(error instanceof Error ? error.message : 'Unable to load this file.')
    }
  }

  useEffect(() => {
    function handleShortcut(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        searchInput.current?.focus()
      }
    }

    window.addEventListener('keydown', handleShortcut)
    return () => window.removeEventListener('keydown', handleShortcut)
  }, [])

  useEffect(() => {
    if (!showImport) return
    function handleEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') setShowImport(false)
    }

    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [showImport])

  useEffect(() => {
    function handleBeforePrint() {
      flushSync(() => {
        setActivePersona('all')
        setSearch('')
        setHideReasoning(false)
      })
    }

    window.addEventListener('beforeprint', handleBeforePrint)
    return () => window.removeEventListener('beforeprint', handleBeforePrint)
  }, [])

  function printContext() {
    flushSync(() => {
      setActivePersona('all')
      setSearch('')
      setHideReasoning(false)
    })
    window.print()
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Icon name="layers" size={19} /></div>
          <div><strong>Context<span>Lens</span></strong><small>AI context explorer</small></div>
        </div>

        <div className="sidebar-section">
          <div className="section-label">WORKSPACE</div>
          <button className="workspace-switcher" onClick={() => { setFileError(''); setShowImport(true) }} aria-label="Import session data">
            <span className="workspace-icon">C</span>
            <span className="workspace-copy"><strong>{filename}</strong><small>Load or paste session JSON</small></span>
            <Icon name="down" size={14} />
          </button>
        </div>

        <div className="sidebar-section persona-section">
          <div className="section-label">PERSONAS <span>{personas.length}</span></div>
          <button className={`persona-item ${activePersona === 'all' ? 'selected' : ''}`} onClick={() => setActivePersona('all')}>
            <span className="persona-icon all-icon"><Icon name="layers" size={15} /></span>
            <span className="persona-copy"><strong>All messages</strong><small>Full conversation</small></span>
            <span className="persona-count">{entries.length}</span>
          </button>
          {personas.map((persona) => (
            <button key={persona.id} className={`persona-item ${activePersona === persona.id ? 'selected' : ''}`} onClick={() => setActivePersona(persona.id)}>
              <span className={`persona-icon ${persona.color}`}><span /></span>
              <span className="persona-copy"><strong>{persona.label}</strong><small>{persona.description}</small></span>
              <span className="persona-count">{personaCounts[persona.id]}</span>
            </button>
          ))}
        </div>

      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="breadcrumbs"><span>Conversations</span><Icon name="chevron" size={13} /><strong>{title}</strong></div>
          <div className="topbar-right">
            {hasContext && <button className="import-button" onClick={printContext}>Print</button>}
            <button className="import-button" onClick={() => { setFileError(''); setShowImport(true) }}>Import session</button>
            <button className="icon-button" aria-label="Toggle tool directory" title="Toggle tool directory" onClick={() => setShowTools((value) => !value)}><Icon name="panel" /></button>
          </div>
        </header>

        <div className={`content-area ${!hasContext ? 'empty-content-area' : ''}`}>
          {!hasContext ? (
            <section className="empty-state welcome-empty-state" aria-labelledby="welcome-title">
              <span className="empty-icon"><Icon name="layers" size={21} /></span>
              <h1 id="welcome-title">Bring your context into view</h1>
              <p>Only context data in the OpenAI API specification format is supported. Provide it to explore messages, tool calls, and assistant responses.</p>
              <button onClick={() => { setFileError(''); setShowImport(true) }}>Provide context data</button>
            </section>
          ) : (
            <>
          <section className="conversation">
            <div className="conversation-heading">
              <div>
                <div className="eyebrow"><span className="eyebrow-dot" /> AI CONTEXT <span className="heading-separator">/</span> {entries.length} MESSAGES</div>
                <h1>{title}</h1>
                <p className="session-subtitle">Explore how the assistant reasoned, responded, and used tools.</p>
              </div>
              {contextData.model && <div className="model-badge"><span className="model-spark">✳</span><span><small>MODEL</small><strong>{contextData.model}</strong></span></div>}
            </div>

            <section className="print-summary" aria-label="Session summary">
              <h2>Session summary</h2>
              <dl className="print-statistics">
                <div><dt>Total messages</dt><dd>{entries.length}</dd></div>
                <div><dt>Different roles</dt><dd>{personas.length}</dd></div>
                <div><dt>Tools available</dt><dd>{tools.length}</dd></div>
              </dl>
              <div className="print-role-breakdown">
                <h3>Role breakdown</h3>
                {personas.map((persona) => (
                  <div key={persona.id}>
                    <span>{persona.label}</span>
                    <strong>{personaCounts[persona.id] ?? 0}</strong>
                  </div>
                ))}
              </div>
              <div className="print-tools">
                <h3>Available tools</h3>
                {tools.length > 0 ? (
                  <ul>
                    {tools.map((tool, index) => (
                      <li key={`${tool.type ?? 'tool'}-${tool.name}-${index}`}>
                        <strong>{tool.name}</strong>
                        <p>{tool.description ?? 'No description provided.'}</p>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p>No tools available.</p>
                )}
              </div>
            </section>

            {instructions && (
              <section className="instructions-card" aria-labelledby="instructions-title">
                <div className="instructions-heading">
                  <span className="instructions-icon"><Icon name="layers" size={15} /></span>
                  <h2 id="instructions-title">Instructions</h2>
                </div>
                <MarkdownContent content={instructions} />
              </section>
            )}

            <details className="response-raw">
              <summary>Full Responses API JSON</summary>
              <pre><code><JsonHighlight text={pretty(contextData.source)} /></code></pre>
            </details>

            <div className="conversation-toolbar">
              <label className="search-field">
                <Icon name="search" size={17} />
                <input ref={searchInput} value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search messages, tools..." aria-label="Search messages" />
                {search && <button className="clear-search" onClick={() => setSearch('')} aria-label="Clear search"><Icon name="close" size={14} /></button>}
                {!search && <kbd>⌘ K</kbd>}
              </label>
              <div className="toolbar-actions">
                <button className={`filter-button reasoning-toggle ${hideReasoning ? 'active' : ''}`} onClick={() => setHideReasoning((value) => !value)}>{hideReasoning ? 'Show reasoning' : 'Hide reasoning'}</button>
              </div>
            </div>

            <div className="stream-label">
              <span>{activePersona === 'all' ? 'CONVERSATION' : `${personas.find((persona) => persona.id === activePersona)?.label.toUpperCase()} MESSAGES`}</span>
              <span>{filteredEntries.length} shown <span className="stream-divider">·</span> in sequence</span>
            </div>

            <div className="message-list">
              {filteredEntries.map((entry) => {
                const persona = personas.find((item) => item.id === entry.persona)
                const isCall = entry.persona === 'function_call'
                const isOutput = entry.persona === 'function_call_output'
                const isReasoning = entry.persona === 'reasoning'
                const isStructuredEntry = !['system', 'developer', 'user', 'assistant', 'reasoning', 'function_call', 'function_call_output'].includes(entry.persona)
                const rawEntry = contextData.input[entry.index]
                const text = isStructuredEntry ? pretty(rawEntry) : entryText(entry)
                return (
                  <article key={entry.id ?? entry.index} className={`message-card ${persona?.color ?? 'unknown'}`}>
                    <div className="message-rail"><span className={`rail-dot ${persona?.color ?? 'unknown'}`} /></div>
                    <div className={`message-inner ${isReasoning ? 'reasoning-inner' : ''} ${isCall || isOutput ? 'tool-inner' : ''}`}>
                      <div className="message-head">
                        <div className="message-identity">
                          <span className={`avatar ${persona?.color ?? 'unknown'}`}>{isCall ? <Icon name="code" size={14} /> : isOutput ? <Icon name="check" size={14} /> : isReasoning ? 'R' : entry.persona === 'system' ? 'S' : entry.persona === 'developer' ? 'D' : entry.persona === 'user' ? 'U' : 'AI'}</span>
                          <strong>{isCall ? entry.name : persona?.label ?? entry.persona}</strong>
                          {isCall && <span className="message-kind">TOOL CALL</span>}
                          {isOutput && <span className="message-kind result-kind">TOOL RESULT</span>}
                          {isReasoning && <span className="message-kind reasoning-kind">INTERNAL</span>}
                        </div>
                        <div className="message-meta">
                          {entry.status && <span className={`status-label ${entry.status}`}>{entry.status}</span>}
                          <span className="message-number">#{String(entry.index + 1).padStart(2, '0')}</span>
                        </div>
                      </div>
                      <div className={`message-content ${isCall || isOutput ? 'code-content' : ''}`}>
                        {isCall && <><div className="tool-summary"><span className="method-chip">{entry.name}</span><span>Requested action</span></div><pre><code><JsonHighlight text={pretty(entry.arguments)} /></code></pre></>}
                        {isOutput && <><div className="tool-summary"><span className="success-dot" /><span>Returned result</span></div><pre><code><JsonHighlight text={text} /></code></pre></>}
                        {!isCall && !isOutput && (
                          isStructuredEntry ? <pre><code><JsonHighlight text={text} /></code></pre> : <MarkdownContent content={text} />
                        )}
                      </div>
                      {entry.call_id && <div className="message-footer"><span>CALL ID</span><code>{entry.call_id}</code></div>}
                      <details className="message-raw">
                        <summary>Raw API item</summary>
                        <pre><code><JsonHighlight text={pretty(rawEntry)} /></code></pre>
                      </details>
                    </div>
                  </article>
                )
              })}
              {filteredEntries.length === 0 && (
                <div className="empty-state">
                  <span className="empty-icon"><Icon name="search" size={21} /></span>
                  <strong>{entries.length === 0 ? 'No conversation messages' : 'No messages found'}</strong>
                  <p>{entries.length === 0 ? 'This context contains no messages.' : 'Try a different search or adjust your filters.'}</p>
                  {entries.length > 0 && <button onClick={() => { setSearch(''); setActivePersona('all'); setHideReasoning(false) }}>Clear all filters</button>}
                </div>
              )}
              {filteredEntries.length > 0 && <div className="end-of-stream"><span /> END OF CONTEXT <span /></div>}
            </div>
          </section>

          <aside className={`detail-panel ${showTools ? 'tools-visible' : ''}`}>
            <div className="detail-heading"><div><span className="detail-overline">SESSION DETAILS</span><h2>At a glance</h2></div><button className="close-details" aria-label="Close session details" onClick={() => setShowTools(false)}><Icon name="close" size={16} /></button></div>
            <div className="detail-card session-card">
              <div className="detail-card-title"><span className="detail-icon"><Icon name="clock" size={15} /></span><strong>Session overview</strong></div>
              <div className="overview-row"><span>Messages</span><strong>{entries.length}</strong></div>
              <div className="overview-row"><span>Personas</span><strong>{personas.filter((persona) => personaCounts[persona.id] > 0).length}</strong></div>
              <div className="overview-row"><span>Tools available</span><strong>{tools.length}</strong></div>
            </div>

            <div className="detail-section">
              <div className="detail-section-heading"><strong>PERSONA BREAKDOWN</strong><span>COUNT</span></div>
              {personas.map((persona) => (
                <button className={`breakdown-row ${activePersona === persona.id ? 'selected' : ''}`} key={persona.id} onClick={() => setActivePersona(activePersona === persona.id ? 'all' : persona.id)}>
                  <span className={`mini-dot ${persona.color}`} /><span>{persona.label}</span><strong>{personaCounts[persona.id] ?? 0}</strong>
                  <span className="breakdown-bar"><i style={{ width: `${(personaCounts[persona.id] / entries.length) * 100}%` }} /></span>
                </button>
              ))}
            </div>

            <div className="detail-section tool-directory">
              <div className="detail-section-heading"><strong>TOOLS IN CONTEXT</strong><span>{tools.length}</span></div>
              <p className="tools-intro">Functions available to the assistant during this session.</p>
              <div className="tool-list">
                {tools.map((tool, index) => (
                  <div className="tool-row" key={`${tool.type ?? 'tool'}-${tool.name}-${index}`}>
                    <span className="tool-symbol"><Icon name="code" size={13} /></span>
                    <span>
                      <strong>{tool.name}</strong>
                      <small>{tool.description?.split('.')[0] ?? `${tool.type ?? 'Context'} tool`}</small>
                      <details className="tool-definition">
                        <summary>View tool specification</summary>
                        <pre><code><JsonHighlight text={JSON.stringify(tool, null, 2)} /></code></pre>
                      </details>
                    </span>
                  </div>
                ))}
              </div>
            </div>

          </aside>
            </>
          )}
        </div>
      </main>
      {showImport && (
        <div className="import-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowImport(false) }}>
          <section className="import-dialog" role="dialog" aria-modal="true" aria-labelledby="import-title">
            <div className="import-dialog-heading">
              <div><span className="detail-overline">SESSION DATA</span><h2 id="import-title">Import a session</h2></div>
              <button className="close-details" aria-label="Close import dialog" onClick={() => setShowImport(false)}><Icon name="close" size={16} /></button>
            </div>
            <p className="import-description">Only context data in the OpenAI API specification format is supported. Load a local JSON file or paste session data below. Your data stays in this browser.</p>
            <button className="file-picker-button" onClick={() => fileInput.current?.click()}>
              <Icon name="layers" size={16} /> Choose a JSON file
            </button>
            <input
              ref={fileInput}
              className="file-input"
              type="file"
              accept=".json,application/json"
              aria-label="Choose a session JSON file"
              onChange={(event) => {
                void loadContextFile(event.currentTarget.files?.[0])
                event.currentTarget.value = ''
              }}
            />
            <div className="import-divider"><span>OR PASTE JSON</span></div>
            <label className="paste-label" htmlFor="session-json">Session JSON</label>
            <textarea
              id="session-json"
              className="json-textarea"
              value={pastedJson}
              onChange={(event) => { setPastedJson(event.target.value); setFileError('') }}
              placeholder={'{\n  "input": [],\n  "tools": []\n}'}
              spellCheck={false}
            />
            {fileError && <p className="file-error import-error" role="alert">{fileError}</p>}
            <div className="import-actions">
              <button className="cancel-import" onClick={() => setShowImport(false)}>Cancel</button>
              <button
                className="submit-import"
                onClick={() => {
                  if (!pastedJson.trim()) {
                    setFileError('Paste session JSON before loading.')
                    return
                  }
                  loadContextJson(pastedJson, 'pasted-session.json')
                }}
              >
                Load session
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  )
}

export default App
